"""
STEP 5 — Propensity score modelling with Naive Bayes  +  IPW-ATE.

Why Naive Bayes here?
  • A propensity model is a CLASSIFIER for P(T = 1 | X).
  • NB is a fast, well-calibrated baseline classifier — perfect for the
    "show the doctor a different family of models" comparison.

We use the propensity scores in the Horvitz–Thompson / IPW estimator
to deconfound the naive ATE.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.naive_bayes    import GaussianNB
from sklearn.linear_model   import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from config import ART, FIG_DIR, TABLE_DIR, RANDOM_STATE, N_BOOTSTRAP
from utils  import banner, sub, save_fig


CATEGORICAL = ["gender", "diagnosis", "icu_unit",
               "admission_type", "bmi_category", "age_group"]


def build_design_matrix(df: pd.DataFrame):
    df = df.copy()
    for c in CATEGORICAL:
        df[c + "_enc"] = LabelEncoder().fit_transform(df[c].astype(str))

    drop = {"patient_id", "scenario", "treatment", "outcome",
            "severity_stratum",
            "admission_time", "discharge_time",
            # ground-truth leakage guards (added by step 03 DGP overlay)
            "true_cate", "base_y",
            "icu_stay_days", "icu_stay_days_raw"} | set(CATEGORICAL)
    feat = [c for c in df.columns if c not in drop and
            df[c].dtype.kind in "biufc"]

    X = df[feat].astype(float).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(X), feat


def main() -> None:
    banner("STEP 5 — NAIVE BAYES PROPENSITY  +  IPW-ATE")
    rng = np.random.default_rng(RANDOM_STATE)
    df  = pd.read_parquet(ART["causal_frame"])

    X, feat = build_design_matrix(df)
    T = df["treatment"].values.astype(int)
    Y = df["outcome"].values.astype(float)

    sub(f"Design matrix : {X.shape}  ({len(feat)} features)")

    # ---------------------------------------------------------- 5.1
    sub("5.1  Fit Gaussian Naive Bayes propensity model")
    nb = GaussianNB().fit(X, T)
    ps = nb.predict_proba(X)[:, 1]           # P(T=1 | X)
    ps = np.clip(ps, 0.05, 0.95)             # numerical safety

    cv_acc = cross_val_score(GaussianNB(), X, T, cv=5,
                             scoring="accuracy")
    cv_auc = cross_val_score(GaussianNB(), X, T, cv=5,
                             scoring="roc_auc")
    print(f"  CV accuracy : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  CV AUC-ROC  : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    print(f"  Propensity range : [{ps.min():.3f}, {ps.max():.3f}]")

    # quick sanity: compare to LogReg propensity (calibration check)
    lr = LogisticRegression(max_iter=2000, n_jobs=-1).fit(X, T)
    ps_lr = np.clip(lr.predict_proba(X)[:, 1], 0.05, 0.95)
    print(f"  LogReg AUC (sanity) : "
          f"{cross_val_score(LogisticRegression(max_iter=2000), X, T, cv=5, scoring='roc_auc').mean():.3f}")

    # ---------------------------------------------------------- 5.2
    sub("5.2  Horvitz–Thompson IPW-ATE")
    def ipw_ate(t, y, p):
        wt = t / p
        wc = (1 - t) / (1 - p)
        return (t * y / p).sum() / wt.sum() - \
               ((1 - t) * y / (1 - p)).sum() / wc.sum()

    ate_ipw_nb = ipw_ate(T, Y, ps)
    ate_ipw_lr = ipw_ate(T, Y, ps_lr)
    naive_ate  = Y[T == 1].mean() - Y[T == 0].mean()

    # bootstrap CI for IPW-NB
    boot = np.empty(N_BOOTSTRAP)
    n = len(Y)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        boot[i] = ipw_ate(T[idx], Y[idx], ps[idx])
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))

    print(f"  Naive ATE                : {naive_ate:+.3f} d")
    print(f"  IPW-ATE (NB propensity)  : {ate_ipw_nb:+.3f} d   "
          f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  IPW-ATE (LogReg ‒ check) : {ate_ipw_lr:+.3f} d")
    print(f"\n  Bias removed by NB-IPW : "
          f"{naive_ate - ate_ipw_nb:+.3f} d")

    # ---------------------------------------------------------- 5.3
    pd.DataFrame([
        {"method": "Naive",                    "ate": naive_ate,
         "ci_lo": np.nan, "ci_hi": np.nan},
        {"method": "IPW (Naive Bayes)",        "ate": ate_ipw_nb,
         "ci_lo": ci[0],  "ci_hi": ci[1]},
        {"method": "IPW (Logistic Regression)","ate": ate_ipw_lr,
         "ci_lo": np.nan, "ci_hi": np.nan},
    ]).to_csv(f"{TABLE_DIR}/ipw_ate.csv", index=False)

    # propensity overlap plot — critical for any IPW analysis
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ps[T == 0], bins=30, alpha=0.6, label="Control (T=0)",
            color="#3498db")
    ax.hist(ps[T == 1], bins=30, alpha=0.6, label="Treated (T=1)",
            color="#e74c3c")
    ax.axvline(0.5, color="black", ls="--")
    ax.set_xlabel("Propensity score  P(T=1 | X)  — Naive Bayes")
    ax.set_ylabel("Patients")
    ax.set_title("Propensity overlap (positivity check)\n"
                 "good overlap → IPW estimates are trustworthy")
    ax.legend()
    save_fig(fig, f"{FIG_DIR}/05a_propensity_overlap.png")

    # persist propensity scores for downstream scripts
    out = pd.DataFrame({"patient_id": df["patient_id"],
                        "propensity_nb": ps})
    out.to_parquet(f"{ART['causal_frame']}".replace(
        ".parquet", "_propensity.parquet"), index=False)


if __name__ == "__main__":
    main()
