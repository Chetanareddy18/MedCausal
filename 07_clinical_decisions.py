"""
STEP 7 — Per-patient clinical decisions  (TREAT / AVOID).

Uses the X-Learner CATE (best for heterogeneous effects) to assign
each of the 500 patients an individualised recommendation.
Adds a 4-quadrant clinical-action map combining causal CATE with NB risk.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from config import ART, TABLE_DIR
from utils  import banner, sub
import importlib
build = importlib.import_module("05_propensity_naive_bayes").build_design_matrix


BEST = "X-Learner (XGBoost)"


def main() -> None:
    banner("STEP 7 — PER-PATIENT CLINICAL DECISIONS")

    df = pd.read_parquet(ART["causal_frame"])
    with open(ART["cate_results"], "rb") as fh:
        bundle = pickle.load(fh)

    cate = bundle["results"][BEST]["cate"]
    df["cate_days"] = cate

    # decision rule: CATE < 0 → treatment is expected to REDUCE ICU stay
    df["recommend_treat"] = (df["cate_days"] < 0).astype(int)
    df["recommendation"]  = np.where(df["recommend_treat"] == 1,
                                     "TREAT — expected benefit",
                                     "AVOID — no expected benefit")

    sub("Decision summary")
    n = len(df); nt = int(df["recommend_treat"].sum())
    print(f"  Recommend TREAT : {nt:>4}   ({nt/n*100:.1f} %)")
    print(f"  Recommend AVOID : {n-nt:>4}   ({(n-nt)/n*100:.1f} %)")

    # ------------------------------------------------------------ NB risk
    sub("Adding Naive Bayes long-stay risk score")
    median_stay = df["outcome"].median()
    Y_bin = (df["outcome"] > median_stay).astype(int).values
    X, _  = build(df)
    nb    = GaussianNB().fit(X, Y_bin)
    df["nb_long_stay_prob"] = nb.predict_proba(X)[:, 1]
    df["nb_high_risk"]      = (df["nb_long_stay_prob"] > 0.5).astype(int)

    def quadrant(row):
        risk  = row["nb_high_risk"] == 1
        helps = row["cate_days"] < 0
        if risk and helps:           return "A · TREAT NOW"
        if risk and not helps:       return "B · INVESTIGATE"
        if not risk and helps:       return "C · PREVENTIVE TREAT"
        return                        "D · MONITOR ONLY"
    df["clinical_action"] = df.apply(quadrant, axis=1)

    sub("Clinical-action quadrants")
    print(df["clinical_action"].value_counts().sort_index())

    # ------------------------------------------------------------ save
    cols = ["patient_id", "age", "gender", "diagnosis", "icu_unit",
            "comorbidity_count", "scenario", "treatment", "outcome",
            "overall_risk_mean", "overall_risk_max",
            "cate_days", "recommend_treat", "recommendation",
            "nb_long_stay_prob", "nb_high_risk", "clinical_action"]
    df[cols].to_parquet(ART["decisions"], index=False)
    df[cols].to_csv(f"{TABLE_DIR}/patient_decisions_500.csv", index=False)

    sub("First 10 patients")
    print(df[cols[:1] + ["age", "diagnosis", "scenario",
                         "outcome", "cate_days",
                         "recommendation",
                         "clinical_action"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
