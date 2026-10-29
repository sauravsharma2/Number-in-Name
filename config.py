from datetime import date

# ─── SOURCE ──────────────────────────────────────────────────────────────────
SRC_SCHEMA = "prd_bronze_orbis.orbis"

# None = all countries in one run. Optional filter for a quick test: ["DK"]
COUNTRIES = None

# ─── MATCHING ────────────────────────────────────────────────────────────────
DIRECT_THRESHOLD       = 96    # Step 3 fuzzy digit cut-off (fuzz.ratio)
MIN_DIGIT_LENGTH       = 7     # Step 3: min digits in name
STEP2_MIN_OVERLAP      = 5     # Step 2 anchored: min digits of the shorter number
STEP2_LENGTH_RATIO_MIN = 0.7   # Step 2 anchored: shorter / longer length

# ─── NAME CHECK (runs on the proposals after Steps 1-3) ──────────────────────
THRESHOLD      = 98                        # MatchScore needed to pass
LEGAL_FORM_CSV = "legal_form_mapping.csv"  # in the project folder
LEGAL_FORM_CHECK = False   # False = different legal forms accepted (forms still removed before scoring)

# ─── OUTPUT ──────────────────────────────────────────────────────────────────
MAX_LINK_IDS = 10      # max IDs listed per link type in Common_Links
OUTPUT_PATH  = f"/Volumes/<catalog>/<schema>/<volume>/Number_in_Name_{date.today():%Y%m%d}.xlsx"

# Intermediate results (Direct, Good, each step). Delta tables are safest for the
# full population; None = Spark localCheckpoint (fine for test runs).
CHECKPOINT_SCHEMA = "hive_metastore.default"

SAVE_DELTA   = False                    # also save results as Delta tables
DELTA_SCHEMA = "hive_metastore.default"  # nin_matches / nin_unmatched
