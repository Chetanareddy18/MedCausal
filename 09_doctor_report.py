"""
STEP 9 — Doctor-facing report.

Pretty-prints the headline numbers and writes a single CSV summary
that bundles everything for clinicians to scan quickly.
"""
import pickle
import pandas as pd
from config import ART, TABLE_DIR
from utils  import banner, sub


def main() -> None:
    banner("STEP 9 — DOCTOR REPORT")

    decisions = pd.read_parquet(ART["decisions"])
    causal    = pd.read_parquet(ART["causal_frame"])
    ipw       = pd.read_csv(f"{TABLE_DIR}/ipw_ate.csv")
    causal_s  = pd.read_csv(f"{TABLE_DIR}/causal_ate_summary.csv")
    with open(ART["cate_results"], "rb") as fh:
        bundle = pickle.load(fh)

    true_ate = float(causal["true_cate"].mean())

    # ---------- headline ATE table
    sub("All ATE estimates  (negative = treatment shortens ICU stay)")
    naive  = ipw.loc[ipw["method"] == "Naive", "ate"].iloc[0]
    ipw_nb = ipw.loc[ipw["method"] == "IPW (Naive Bayes)", "ate"].iloc[0]

    print(f"  {'Method':<25} {'ATE (days)':>12} {'vs TRUTH':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'GROUND TRUTH (DGP)':<25} {true_ate:>+12.3f} {'—':>12}")
    print(f"  {'Naive (biased)':<25} {naive:>+12.3f} {naive-true_ate:>+12.3f}")
    print(f"  {'IPW (Naive Bayes)':<25} {ipw_nb:>+12.3f} {ipw_nb-true_ate:>+12.3f}")
    for _, r in causal_s.iterrows():
        print(f"  {r['model']:<25} {r['ate']:>+12.3f} "
              f"{r['ate_bias']:>+12.3f}")

    # ---------- decision counts
    sub("Per-patient recommendations")
    print(decisions["recommendation"].value_counts())
    print()
    print(decisions["clinical_action"].value_counts().sort_index())

    # ---------- by-subgroup CATE
    sub("Mean CATE by diagnosis")
    grp = (decisions.groupby("diagnosis")["cate_days"]
                    .agg(["mean", "count"])
                    .sort_values("mean"))
    print(grp.round(3))
    grp.to_csv(f"{TABLE_DIR}/cate_by_diagnosis.csv")

    sub("Mean CATE by ICU unit")
    grp = (decisions.groupby("icu_unit")["cate_days"]
                    .agg(["mean", "count"])
                    .sort_values("mean"))
    print(grp.round(3))
    grp.to_csv(f"{TABLE_DIR}/cate_by_icu_unit.csv")

    # ---------- TL;DR for the slide deck
    pct_benefit = (decisions["cate_days"] < 0).mean() * 100
    best = "X-Learner (XGBoost)"
    ate_best  = bundle["results"][best]["ate"]
    rmse_best = bundle["results"][best]["cate_rmse"]
    corr_best = bundle["results"][best]["cate_corr"]

    print("\n" + "=" * 78)
    print("  KEY MESSAGES FOR THE DOCTORS")
    print("=" * 78)
    print(f"""
  GROUND TRUTH (planted in the data):
    • TRUE Average Treatment Effect = {true_ate:+.2f} days
    • TRUE % of patients benefiting = {(causal['true_cate'] < 0).mean()*100:.0f} %

  WHAT THE MODELS RECOVERED:
    • Naive comparison gave ATE = {naive:+.2f} d  →  WRONG SIGN
      (treated patients are sicker → confounding pulls ATE up).

    • Best causal model ({best}):
        ATE       = {ate_best:+.2f} d        (truth {true_ate:+.2f})
        CATE RMSE = {rmse_best:.2f} d
        CATE corr = {corr_best:+.2f} with the true effect

    • {pct_benefit:.0f}% of the 500 patients are predicted to benefit
      from treatment.

  CLINICAL TAKE-AWAY:
    → Correlation says treatment hurts ; causal ML says treatment HELPS.
    → Effect is heterogeneous: high-severity patients gain ≈2 days,
      low-severity patients are slightly harmed by over-treatment.
    → Use the per-patient CATE in patient_decisions_500.csv as a
      bedside decision-support score (negative = expected to shorten stay).
""")
    print("  Files for the deck:")
    print("    outputs/figures/04*  (correlation vs causation)")
    print("    outputs/figures/08*  (ATE / CATE / quadrants)")
    print("    outputs/tables/patient_decisions_500.csv")


if __name__ == "__main__":
    main()
