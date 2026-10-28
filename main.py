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
importlib.reload(config)
importlib.reload(matcher)


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

unmatched = materialize(
    direct.join(matches.select(F.col("Direct_ID").alias("Id")), "Id", "left_anti")
          .select(F.col("Id").alias("Direct_ID"), F.col("Name").alias("Direct_Name"),
                  F.col("Digits").alias("Extracted_Digits"),
                  F.col("Information_provider").alias("Direct_Information_Provider")),
    "Unmatched")

display(matches.groupBy("Match_Type")
               .agg(F.count("*").alias("matches"),
                    F.sum(F.when(F.col("Common_Links") != "No", 1).otherwise(0)).alias("with_common_links"))
               .orderBy("Match_Type"))
print(f"Total {time.time() - t0:.1f}s")

# COMMAND ----------

# ─── OUTPUT ──────────────────────────────────────────────────────────────────
matcher.write_excel([
    ("Direct Matches",   matches.where("Match_Type = 'Direct'"),   matcher.MATCH_COLS),
    ("Anchored Matches", matches.where("Match_Type = 'Anchored'"), matcher.MATCH_COLS),
    ("Fuzzy Matches",    matches.where("Match_Type = 'Fuzzy'"),    matcher.MATCH_COLS),
    ("Unmatched",        unmatched,                                matcher.UNMATCHED_COLS),
], config.OUTPUT_PATH)

if config.SAVE_DELTA:
    matches.write.mode("overwrite").option("overwriteSchema", "true") \
           .saveAsTable(f"{config.DELTA_SCHEMA}.nin_matches")
    unmatched.write.mode("overwrite").option("overwriteSchema", "true") \
             .saveAsTable(f"{config.DELTA_SCHEMA}.nin_unmatched")
    print(f"Delta saved to {config.DELTA_SCHEMA}")
