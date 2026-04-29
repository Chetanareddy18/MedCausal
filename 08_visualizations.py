"""
STEP 8 — Visualisations for the doctor presentation.

Reads cached results from steps 4–7 and writes publication-ready PNGs
into outputs/figures/. Six panels:
  08a  ATE comparison ladder  (naive → IPW → S → T → X → R)
  08b  CATE distributions per learner
  08c  CATE vs severity scatter
  08d  CATE by risk group (violin)
  08e  Top-15 features driving CATE  (XGBoost importance proxy)
  08f  Decision quadrant map  (NB risk × CATE)
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

from config import ART, FIG_DIR, TABLE_DIR, XGB_PARAMS
from utils  import banner, sub, save_fig


def main() -> None:
    banner("STEP 8 — VISUALISATIONS")

    df       = pd.read_parquet(ART["decisions"])
    causal   = pd.read_parquet(ART["causal_frame"])
    with open(ART["cate_results"], "rb") as fh:
        bundle = pickle.load(fh)
    results = bundle["results"]
    X, feat = bundle["X"], bundle["feature_names"]

    # --- naive + IPW values from previous CSVs
    ipw   = pd.read_csv(f"{TABLE_DIR}/ipw_ate.csv")
    naive = float(ipw.loc[ipw["method"] == "Naive", "ate"].iloc[0])
    ipw_nb = float(ipw.loc[ipw["method"] == "IPW (Naive Bayes)",
                           "ate"].iloc[0])

    # ============================================================ 08a
    sub("08a · ATE comparison ladder")
    methods = ["Naive (biased)", "IPW (Naive Bayes)"] + list(results)
    ates    = [naive, ipw_nb] + [r["ate"] for r in results.values()]
    los     = [np.nan, ipw["ci_lo"][1]] + [r["lo"] for r in results.values()]
    his     = [np.nan, ipw["ci_hi"][1]] + [r["hi"] for r in results.values()]
    colors  = ["#e74c3c", "#f39c12"] + ["#3498db"] * len(results)

    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(methods))
    err = [
        [a - lo if not np.isnan(lo) else 0 for a, lo in zip(ates, los)],
        [hi - a if not np.isnan(hi) else 0 for a, hi in zip(ates, his)],
    ]
    ax.barh(y, ates, xerr=err, color=colors, alpha=0.85,
            edgecolor="white", capsize=5)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(methods)
    ax.set_xlabel("Average Treatment Effect on ICU stay (days)")
    ax.set_title("ATE estimates — naive vs propensity vs causal ML\n"
                 "negative = treatment reduces ICU stay")
    for i, v in enumerate(ates):
        ax.text(v + 0.02, i, f"{v:+.2f}", va="center")
    save_fig(fig, f"{FIG_DIR}/08a_ate_comparison.png")

    # ============================================================ 08b
    sub("08b · CATE distributions per learner")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (name, r) in zip(axes.flatten(), results.items()):
        c = r["cate"]
        ax.hist(c, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", ls="--", label="No effect")
        ax.axvline(c.mean(), color="green",
                   label=f"Mean {c.mean():+.3f}")
        ax.set_title(name); ax.set_xlabel("CATE (days)")
        ax.legend(fontsize=9)
    fig.suptitle("Individual treatment effects (CATE) — heterogeneity",
                 fontweight="bold")
    save_fig(fig, f"{FIG_DIR}/08b_cate_distributions.png")

    # ============================================================ 08c
    sub("08c · CATE vs severity")
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(causal["overall_risk_mean"], df["cate_days"],
                    c=df["treatment"], cmap="coolwarm", alpha=0.6, s=25)
    ax.axhline(0, color="black", ls="--")
    ax.set_xlabel("Overall risk score (mean)")
    ax.set_ylabel("CATE — days saved by treatment")
    ax.set_title("Treatment effect heterogeneity by patient severity")
    plt.colorbar(sc, label="Actual treatment (0/1)")
    save_fig(fig, f"{FIG_DIR}/08c_cate_vs_severity.png")

    # ============================================================ 08d
    sub("08d · CATE by risk tertile")
    tmp = df.copy()
    tmp["risk_group"] = pd.qcut(causal["overall_risk_mean"], 3,
                                labels=["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=tmp, x="risk_group", y="cate_days",
                   palette="RdYlGn_r", order=["Low", "Medium", "High"], ax=ax)
    ax.axhline(0, color="black", ls="--")
    ax.set_ylabel("CATE (days)")
    ax.set_title("Treatment effect by patient risk group\n"
                 "below zero = treatment is beneficial")
    save_fig(fig, f"{FIG_DIR}/08d_cate_by_risk_group.png")

    # ============================================================ 08e
    sub("08e · Drivers of treatment-effect heterogeneity")
    cate = results["X-Learner (XGBoost)"]["cate"]
    fi = XGBRegressor(**XGB_PARAMS).fit(X, cate)
    imp = (pd.DataFrame({"feature": feat, "importance": fi.feature_importances_})
             .sort_values("importance", ascending=True).tail(15))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp["feature"], imp["importance"], color="steelblue")
    ax.set_xlabel("Feature importance (XGB on CATE)")
    ax.set_title("Top 15 features driving WHO benefits from treatment")
    save_fig(fig, f"{FIG_DIR}/08e_cate_drivers.png")

    # ============================================================ 08f
    sub("08f · Decision quadrant map")
    palette = {"A · TREAT NOW":          "#e74c3c",
               "B · INVESTIGATE":        "#f39c12",
               "C · PREVENTIVE TREAT":   "#2ecc71",
               "D · MONITOR ONLY":       "#3498db"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for q, c in palette.items():
        m = df["clinical_action"] == q
        ax.scatter(df.loc[m, "cate_days"], df.loc[m, "nb_long_stay_prob"],
                   color=c, label=q, alpha=0.6, s=28)
    ax.axhline(0.5, color="gray", ls="--")
    ax.axvline(0,   color="gray", ls="-")
    ax.set_xlabel("CATE — causal effect on ICU stay (days)")
    ax.set_ylabel("Naive Bayes  P(long stay)")
    ax.set_title("Bedside decision map  —  combining risk × causal effect")
    ax.legend(loc="upper right", fontsize=9)
    save_fig(fig, f"{FIG_DIR}/08f_decision_quadrants.png")

    # ============================================================ 08g
    sub("08g · Recovered CATE vs ground-truth τ")
    true_cate = causal["true_cate"].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: scatter recovered vs true (best learner)
    best = "X-Learner (XGBoost)"
    cate_best = results[best]["cate"]
    axes[0].scatter(true_cate, cate_best, alpha=0.5, s=22, color="#2c3e50")
    lo, hi = float(min(true_cate.min(), cate_best.min())), \
             float(max(true_cate.max(), cate_best.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect recovery")
    r = np.corrcoef(true_cate, cate_best)[0, 1]
    rmse = float(np.sqrt(np.mean((cate_best - true_cate) ** 2)))
    axes[0].set_xlabel("TRUE τ (planted in DGP)")
    axes[0].set_ylabel(f"Recovered CATE — {best}")
    axes[0].set_title(f"Recovery scatter   r={r:+.2f}   RMSE={rmse:.2f} d")
    axes[0].legend()

    # right: bar — bias of each learner
    names = list(results)
    biases = [results[n]["ate"] - true_cate.mean() for n in names]
    colors = ["#2ecc71" if abs(b) < 0.5 else "#e74c3c" for b in biases]
    axes[1].barh(names, biases, color=colors, alpha=0.85)
    axes[1].axvline(0, color="black")
    axes[1].set_xlabel("ATE bias (estimate − truth)  in days")
    axes[1].set_title(f"ATE bias vs ground truth  (TRUE ATE = "
                      f"{true_cate.mean():+.2f} d)")
    save_fig(fig, f"{FIG_DIR}/08g_recovery_vs_truth.png")


if __name__ == "__main__":
    main()
