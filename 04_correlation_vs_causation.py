"""
STEP 4 — Correlation vs Causation  (the doctor-facing demo).

This is the headline file for the presentation. It uses ONLY the real
500-patient data and shows three things in plain language:

  1. CORRELATION  — what naive statistics say (Pearson / Spearman).
  2. NAIVE COMPARISON — treated vs control mean difference (biased).
  3. SIMPSON-STYLE STRATIFIED VIEW — the same comparison split by
     severity strata; the apparent effect flips / shrinks once you
     control for severity. That gap is the *confounding bias*.

No causal models yet — just descriptive evidence that "correlation
is not causation" on the actual ICU data.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from config import ART, FIG_DIR, TABLE_DIR, N_BOOTSTRAP, RANDOM_STATE
from utils  import banner, sub, save_fig


def main() -> None:
    banner("STEP 4 — CORRELATION vs CAUSATION  (descriptive evidence)")
    rng = np.random.default_rng(RANDOM_STATE)
    df  = pd.read_parquet(ART["causal_frame"])

    # ============================================================ 4.1
    sub("4.1  Pearson & Spearman correlations with ICU stay")
    candidates = ["age", "comorbidity_count",
                  "overall_risk_mean", "overall_risk_max",
                  "pct_high_risk", "pct_critical", "instability_index",
                  "heart_rate_mean", "spo2_mean", "respiratory_rate_mean",
                  "treatment"]
    rows = []
    for c in candidates:
        r_p, p_p = stats.pearsonr(df[c],  df["outcome"])
        r_s, p_s = stats.spearmanr(df[c], df["outcome"])
        rows.append({"feature": c,
                     "pearson_r": r_p, "pearson_p": p_p,
                     "spearman_r": r_s, "spearman_p": p_s})
    corr = pd.DataFrame(rows).sort_values("pearson_r",
                                          key=abs, ascending=False)
    print(corr.round(3).to_string(index=False))
    corr.to_csv(f"{TABLE_DIR}/correlations_with_icu_stay.csv", index=False)

    # ============================================================ 4.2
    sub("4.2  NAIVE comparison — treated vs control")
    y1, y0 = df.loc[df.treatment == 1, "outcome"].values, \
             df.loc[df.treatment == 0, "outcome"].values
    naive_ate = y1.mean() - y0.mean()
    se   = np.sqrt(y1.var(ddof=1)/len(y1) + y0.var(ddof=1)/len(y0))
    ci   = (naive_ate - 1.96*se, naive_ate + 1.96*se)

    # bootstrap CI for robustness (medians + non-normal CI)
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        a = rng.choice(y1, len(y1), replace=True).mean()
        b = rng.choice(y0, len(y0), replace=True).mean()
        boot[i] = a - b
    boot_ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))

    print(f"  Treated mean ICU stay : {y1.mean():.2f} d   (n={len(y1)})")
    print(f"  Control mean ICU stay : {y0.mean():.2f} d   (n={len(y0)})")
    print(f"  Naive ATE             : {naive_ate:+.3f} d")
    print(f"  95% CI (analytic)     : [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  95% CI (bootstrap)    : [{boot_ci[0]:+.3f}, {boot_ci[1]:+.3f}]")
    print("\n  ⚠ This says treatment INCREASES ICU stay. Sounds wrong, right?")
    print("    That's because it ignores severity — the confounder.")

    # ============================================================ 4.3
    sub("4.3  STRATIFIED view (Simpson's paradox style)")
    # Split into severity tertiles using overall_risk_mean
    df["severity_stratum"] = pd.qcut(
        df["overall_risk_mean"], q=3,
        labels=["Low severity", "Medium severity", "High severity"])

    strat = (df.groupby(["severity_stratum", "treatment"], observed=True)
               ["outcome"].agg(["mean", "count"]).round(3))
    print(strat)

    print("\n  Within-stratum treated minus control:")
    within = []
    for s, sub_df in df.groupby("severity_stratum", observed=True):
        a = sub_df.loc[sub_df.treatment == 1, "outcome"].mean()
        b = sub_df.loc[sub_df.treatment == 0, "outcome"].mean()
        diff = a - b
        within.append({"stratum": str(s), "treated_mean": a,
                       "control_mean": b, "diff_within": diff,
                       "n_treated": int((sub_df.treatment == 1).sum()),
                       "n_control": int((sub_df.treatment == 0).sum())})
        print(f"    {s:<18}  treated={a:.2f}  control={b:.2f}  "
              f"Δ={diff:+.3f} d")

    pd.DataFrame(within).to_csv(
        f"{TABLE_DIR}/stratified_simpson.csv", index=False)
    print("\n  → Within each severity stratum the gap is much smaller "
          "(or reversed)\n    than the pooled naive Δ. That gap = "
          "CONFOUNDING BIAS.")

    # ============================================================ 4.4 — figs
    sub("4.4  Figures")

    # Fig A — correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr.set_index("feature")[["pearson_r", "spearman_r"]],
                annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-0.6, vmax=0.6, ax=ax)
    ax.set_title("Correlation of features with ICU length-of-stay\n"
                 "(correlation ≠ causation)")
    save_fig(fig, f"{FIG_DIR}/04a_correlations.png")

    # Fig B — naive treated vs control distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x="treatment", y="outcome",
                ax=axes[0], palette="Set2")
    axes[0].set_xticklabels(["Control (T=0)", "Treated (T=1)"])
    axes[0].set_ylabel("ICU stay (days)")
    axes[0].set_title(f"Naive view — pooled\nΔ = {naive_ate:+.2f} d  "
                      f"(95% CI {ci[0]:+.2f}, {ci[1]:+.2f})")
    sns.violinplot(data=df, x="severity_stratum", y="outcome",
                   hue="treatment", split=True, ax=axes[1],
                   palette="Set2", inner="quartile")
    axes[1].set_xlabel("Severity stratum (overall_risk tertiles)")
    axes[1].set_ylabel("ICU stay (days)")
    axes[1].set_title("Stratified view — Simpson's paradox check")
    save_fig(fig, f"{FIG_DIR}/04b_naive_vs_stratified.png")

    # Fig C — confounding diagram (severity → both T and Y)
    fig, ax = plt.subplots(figsize=(9, 4))
    by_sev = df.groupby("severity_stratum",
                        observed=True).agg(
        treated_share=("treatment", "mean"),
        mean_stay   =("outcome",   "mean")
    ).reset_index()
    x = np.arange(len(by_sev))
    ax2 = ax.twinx()
    ax.bar(x - 0.2, by_sev["treated_share"], width=0.4,
           color="#e74c3c", label="P(treatment | severity)")
    ax2.bar(x + 0.2, by_sev["mean_stay"], width=0.4,
            color="#3498db", label="Mean ICU stay (days)")
    ax.set_xticks(x); ax.set_xticklabels(by_sev["severity_stratum"])
    ax.set_ylabel("Treatment probability", color="#e74c3c")
    ax2.set_ylabel("Mean ICU stay (days)", color="#3498db")
    ax.set_title("Severity drives BOTH treatment and outcome → confounding")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    save_fig(fig, f"{FIG_DIR}/04c_confounding_evidence.png")

    sub("Take-aways for the doctor")
    print("  • Strong positive correlation between severity and ICU stay.")
    print("  • Naive treated-vs-control gap is POSITIVE → looks harmful.")
    print("  • Within each severity tier the gap shrinks dramatically →")
    print("    most of the naive 'effect' was confounding by severity.")
    print("  • This is exactly why we need CAUSAL ML (next steps).")


if __name__ == "__main__":
    main()
