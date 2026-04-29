"""
Central configuration for the ICU Causal ML pipeline (500 patients).
Edit constants here; every other script imports from this file.
"""

import os
import warnings

# Silence noisy sklearn / causalml deprecation chatter for a clean demo log.
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# -------------------------------------------------------------- paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIG_DIR     = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR   = os.path.join(OUTPUT_DIR, "tables")
CACHE_DIR   = os.path.join(OUTPUT_DIR, "_cache")

for d in (OUTPUT_DIR, FIG_DIR, TABLE_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

VITALS_CSV   = os.path.join(DATA_DIR, "patient_vitals.csv")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients_meta.csv")

# -------------------------------------------------------------- causal framework
# Treatment = active clinical intervention (binary).
# Doctors recognise these scenarios as "the team intervened".
TREATMENT_SCENARIOS = {
    "fever_infection",          # antibiotics + cooling
    "hypoxic_deterioration",    # oxygen support / NIV / intubation
    "hemodynamic_instability",  # vasopressors / fluids
    "septic_shock",             # bundle: antibiotics + fluids + pressors
}

# Outcome
OUTCOME_PRIMARY   = "icu_stay_days"     # continuous (days) — for ATE / CATE
OUTCOME_SECONDARY = "ever_critical"     # binary (max overall_risk >= 3)

# -------------------------------------------------------------- modelling
RANDOM_STATE = 42
N_BOOTSTRAP  = 1000      # for CIs on naive ATE / IPW
XGB_PARAMS   = dict(n_estimators=200, max_depth=4,
                    learning_rate=0.05, random_state=RANDOM_STATE,
                    n_jobs=-1, verbosity=0)

# -------------------------------------------------------------- artefacts
# Scripts persist intermediate results here so each can run standalone.
ART = dict(
    features      = os.path.join(CACHE_DIR, "features.parquet"),
    causal_frame  = os.path.join(CACHE_DIR, "causal_frame.parquet"),
    cate_results  = os.path.join(CACHE_DIR, "cate_results.pkl"),
    decisions     = os.path.join(CACHE_DIR, "decisions.parquet"),
)
