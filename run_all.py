"""
Run the entire ICU Causal ML pipeline end-to-end.

Usage (from CausalML/icu_causal_500):
    python run_all.py
"""
import importlib
import time

STEPS = [
    ("01_load_data",                  "Load & sanity check raw data"),
    ("02_feature_engineering",        "Aggregate vitals → patient features"),
    ("03_treatment_outcome",          "Define treatment & outcome"),
    ("04_correlation_vs_causation",   "Correlation vs causation demo"),
    ("05_propensity_naive_bayes",     "Naive Bayes propensity + IPW-ATE"),
    ("06_causal_meta_learners",       "S / T / X learners → ATE + CATE"),
    ("07_clinical_decisions",         "Per-patient decisions + quadrants"),
    ("08_visualizations",             "Doctor-facing figures"),
    ("09_doctor_report",              "Final printed summary"),
    ("10_bootstrap_cate_ci",          "Bootstrap 95% CI per patient"),
    ("11_subgroup_discovery",         "Tree-based subgroup discovery"),
    ("12_sensitivity_evalue",         "Sensitivity analysis + E-value"),
    ("13_dr_learner",                 "DR-Learner with 5-fold cross-fitting"),
    ("14_shap_explain",               "SHAP per-patient explanations"),
]


def main() -> None:
    t0 = time.time()
    for mod_name, label in STEPS:
        print(f"\n\n{'#' * 78}\n#  {label}\n{'#' * 78}")
        mod = importlib.import_module(mod_name)
        mod.main()
    print(f"\n\nPipeline finished in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
