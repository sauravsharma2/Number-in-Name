# Databricks notebook source
# MAGIC %md
# MAGIC # Number in Name – all countries, one run (PySpark)
# MAGIC SQL (`sql/*.sql`) only fetches data. All matching is in `matcher.py` and runs distributed.
# MAGIC Matches are only searched within the same country (first 2 chars of the ID).
# MAGIC
# MAGIC | Step | Sheet | Rule (digits only) |
# MAGIC |---|---|---|
# MAGIC | 1 | Direct Matches | number in name = Good ID digits |
# MAGIC | 2 | Anchored Matches | one number is the start/end of the other |
# MAGIC | 3 | Fuzzy Matches | fuzz.ratio >= 96 on leading-zero-stripped digits |
# MAGIC | – | Unmatched | nothing found |

# COMMAND ----------

# MAGIC %pip install openpyxl rapidfuzz unidecode

# COMMAND ----------

import importlib
import os
import sys
import time
from string import Template

from pyspark.sql import functions as F

PROJECT_DIR = os.getcwd()                 # folder with main.py, matcher.py, config.py, sql/
sys.path.insert(0, PROJECT_DIR)

import config
import matcher
import namecheck
importlib.reload(config)
importlib.reload(matcher)
importlib.reload(namecheck)


def run_sql(file_name, **params):
    """Read sql/<file_name>, fill ${placeholders}, return a Spark DataFrame."""
    with open(os.path.join(PROJECT_DIR, "sql", file_name), encoding="utf-8") as f:
        sql = Template(f.read()).substitute(src=config.SRC_SCHEMA, **params)
    return spark.sql(sql)


def materialize(df, name):
    """
    Compute once and cut the lineage so later steps don't recompute / re-plan the chain.
    CHECKPOINT_SCHEMA set  -> written as Delta table nin_tmp_<name> (best for full population)
    CHECKPOINT_SCHEMA None -> Spark localCheckpoint (executor disk)
    """
    t = time.time()
    if config.CHECKPOINT_SCHEMA:
        tbl = f"{config.CHECKPOINT_SCHEMA}.nin_tmp_{name.lower().replace(' ', '_')}"
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl)
        df = spark.table(tbl)
    else:
        df = df.localCheckpoint(eager=True)
    n = df.count()
    print(f"  {name:<10} {n:>12,}  ({time.time() - t:.1f}s)")
    return df

# COMMAND ----------

# ─── FETCH + PREPARE ─────────────────────────────────────────────────────────
t0 = time.time()
direct = materialize(matcher.prepare_direct(run_sql("direct.sql"), config.COUNTRIES), "Direct")
good   = materialize(matcher.prepare_good(run_sql("good.sql"), direct), "Good")

# COMMAND ----------

# ─── MATCHING ────────────────────────────────────────────────────────────────
s1 = materialize(matcher.step1_exact(direct, good), "Step 1")

rem = direct.join(s1.select(F.col("Direct_ID").alias("Id")), "Id", "left_anti")
s2 = materialize(matcher.step2_anchored(rem, good, config.STEP2_MIN_OVERLAP,
                                        config.STEP2_LENGTH_RATIO_MIN), "Step 2")

rem = rem.join(s2.select(F.col("Direct_ID").alias("Id")), "Id", "left_anti")
s3 = materialize(matcher.step3_fuzzy(rem, good, config.DIRECT_THRESHOLD,
                                     config.MIN_DIGIT_LENGTH), "Step 3")

# COMMAND ----------

# ─── COMMON LINKS + NAMES ────────────────────────────────────────────────────
matches = (s1.withColumn("Score", F.lit(None).cast("double"))
             .unionByName(s2.withColumn("Score", F.lit(None).cast("double")))
             .unionByName(s3))

matches = matcher.add_common_links(matches, run_sql("links.sql"), config.MAX_LINK_IDS)

d_info = direct.select(F.col("Id").alias("Direct_ID"), F.col("Name").alias("Direct_Name"),
                       F.col("Digits").alias("Direct_Digits"),
                       F.col("Information_provider").alias("Direct_Information_Provider"))
g_info = good.groupBy("Good_ID").agg(F.first("Good_Name", ignorenulls=True).alias("Good_Name"))

matches = materialize(matches.join(d_info, "Direct_ID").join(g_info, "Good_ID"), "Matches")

no_proposal = materialize(
    direct.join(matches.select(F.col("Direct_ID").alias("Id")), "Id", "left_anti")
          .select(F.col("Id").alias("Direct_ID"), F.col("Name").alias("Direct_Name"),
                  F.col("Digits").alias("Extracted_Digits"),
                  F.col("Information_provider").alias("Direct_Information_Provider")),
    "No digit match")

display(matches.groupBy("Match_Type")
               .agg(F.count("*").alias("proposals"),
                    F.sum(F.when(F.col("Common_Links") != "No", 1).otherwise(0)).alias("with_common_links"))
               .orderBy("Match_Type"))
print(f"Total {time.time() - t0:.1f}s")

# COMMAND ----------

# ─── NAME CHECK: current Good name, then Good previous names ────────────────
import numpy as np
import pandas as pd

t = time.time()
lf = namecheck.load_legal_forms(os.path.join(PROJECT_DIR, config.LEGAL_FORM_CSV))

# 1) current name
pdf = namecheck.check_pairs(matches.toPandas(), lf, config.THRESHOLD, config.LEGAL_FORM_CHECK)

# 2) previous names of the Good ID – only for pairs that failed on the current name
failed_ids = pdf.loc[~pdf["Name_Pass"], "Good_ID"].dropna().unique().tolist()
prev_map = {}
if failed_ids:
    ids_df = spark.createDataFrame([(x,) for x in failed_ids], "gid string")
    prev_pdf = (run_sql("prev_names.sql")
                .join(ids_df, F.col("Id") == F.col("gid"), "left_semi")
                .toPandas())
    prev_map = namecheck.build_prev_map(prev_pdf)
pdf = namecheck.check_previous(pdf, prev_map, lf, config.THRESHOLD, config.LEGAL_FORM_CHECK)

pdf["Stage"] = np.where(pdf["Name_Pass"], "Current",
                np.where(pdf["Prev_Pass"], "Previous", "Rejected"))
print(f"Name check {time.time() - t:.1f}s")
print(pdf.groupby(["Match_Type", "Stage"]).size().to_string())

# COMMAND ----------

# ─── BUILD SHEETS ────────────────────────────────────────────────────────────
def with_count(df):
    """Direct_ID_Count = how many times the Direct_ID appears in THIS sheet."""
    df = df.copy()
    df["Direct_ID_Count"] = df.groupby("Direct_ID")["Direct_ID"].transform("size")
    return df

cur  = pdf[pdf["Stage"] == "Current"]
prev = pdf[pdf["Stage"] == "Previous"].copy()
prev["Name_Score"] = prev["Prev_Score"]          # score of the matching previous name

# Unmatched = Direct IDs with NO accepted pair (current or previous) + no digit match at all
accepted = set(pdf.loc[pdf["Stage"] != "Rejected", "Direct_ID"])
rej = pdf[(pdf["Stage"] == "Rejected") & (~pdf["Direct_ID"].isin(accepted))]
rej_rows = pd.DataFrame({
    "Direct_ID": rej["Direct_ID"], "Direct_Name": rej["Direct_Name"],
    "Extracted_Digits": rej["Direct_Digits"],
    "Direct_Information_Provider": rej["Direct_Information_Provider"],
    "Rejected_Good_ID": rej["Good_ID"], "Rejected_Good_Name": rej["Good_Name"],
    "Fail_Reason": ("Current: " + rej["Fail_Reason"].fillna("")
                    + " | Previous: " + rej["Prev_Fail_Reason"].fillna("")),
})
nodig = no_proposal.toPandas()
nodig["Fail_Reason"] = "No digit match"
unmatched = pd.concat([rej_rows, nodig], ignore_index=True)

sheets = [
    ("Direct Matches",        with_count(cur[cur["Match_Type"] == "Direct"]),     matcher.MATCH_COLS),
    ("Anchored Matches",      with_count(cur[cur["Match_Type"] == "Anchored"]),   matcher.MATCH_COLS),
    ("Fuzzy Matches",         with_count(cur[cur["Match_Type"] == "Fuzzy"]),      matcher.MATCH_COLS),
    ("Direct_previous_name",  with_count(prev[prev["Match_Type"] == "Direct"]),   matcher.PREV_COLS),
    ("Anchored_previous_name",with_count(prev[prev["Match_Type"] == "Anchored"]), matcher.PREV_COLS),
    ("Fuzzy_previous_name",   with_count(prev[prev["Match_Type"] == "Fuzzy"]),    matcher.PREV_COLS),
    ("Unmatched",             unmatched,                                          matcher.UNMATCHED_COLS),
]
for title, df, _ in sheets:
    print(f"  {title:<24} {len(df):>10,}")

# COMMAND ----------

# ─── OUTPUT ──────────────────────────────────────────────────────────────────
matcher.write_excel(sheets, config.OUTPUT_PATH)

if config.SAVE_DELTA:
    for name, df in [("nin_proposals", pdf), ("nin_unmatched", unmatched)]:
        sdf = spark.createDataFrame(df.astype(str).replace({"None": None, "nan": None}))
        sdf.write.mode("overwrite").option("overwriteSchema", "true") \
           .saveAsTable(f"{config.DELTA_SCHEMA}.{name}")
    print(f"Delta saved to {config.DELTA_SCHEMA}")
