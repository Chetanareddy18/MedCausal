"""
STEP 11 — Subgroup discovery on CATE.

Idea: fit a SHALLOW decision tree where the TARGET is the per-patient
CATE.  The tree's leaves automatically describe interpretable subgroups
like "age > 70  AND  spo2_mean < 92  →  expected benefit −2.1 days".

This turns a black-box ML model into bedside rules a doctor can act on.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
import matplotlib.pyplot as plt

from config import ART, FIG_DIR, TABLE_DIR
from utils  import banner, sub, save_fig


def main() -> None:
    banner("STEP 11 — SUBGROUP DISCOVERY  (interpretable rules from CATE)")
    with open(ART["cate_results"], "rb") as fh:
        bundle = pickle.load(fh)
    cate = bundle["results"]["X-Learner (XGBoost)"]["cate"]
    feat = bundle["feature_names"]
    X    = bundle["X"]

    sub("Fitting depth-3 decision tree on CATE (X-Learner)")
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=25,
                                 random_state=42)
    tree.fit(X, cate)
    pred = tree.predict(X)
    r2 = float(np.corrcoef(pred, cate)[0, 1] ** 2)
    print(f"  Tree explains R^2 = {r2:.2f} of CATE variance")

    print("\n  ASCII rule list:")
    print(export_text(tree, feature_names=feat, decimals=2))

    # ----- per-leaf summary
    leaf_id = tree.apply(X)
    df = pd.read_parquet(ART["causal_frame"])
    df_leaf = pd.DataFrame({
        "leaf": leaf_id, "cate": cate,
        "true_cate": df["true_cate"].values,
        "treatment": df["treatment"].values,
        "scenario":  df["scenario"].values,
        "diagnosis": df["diagnosis"].values,
        "age":       df["age"].values,
        "risk":      df["overall_risk_mean"].values,
    })
    summary = (df_leaf.groupby("leaf")
               .agg(n=("cate", "size"),
                    mean_cate=("cate", "mean"),
                    mean_true=("true_cate", "mean"),
                    mean_age=("age", "mean"),
                    mean_risk=("risk", "mean"),
                    pct_treated=("treatment", "mean"),
                    top_diagnosis=("diagnosis",
                                   lambda s: s.value_counts().index[0]))
               .sort_values("mean_cate")
               .reset_index())
    summary["recommendation"] = np.where(
        summary["mean_cate"] < -0.3, "STRONG TREAT",
        np.where(summary["mean_cate"] < 0,  "TREAT",
        np.where(summary["mean_cate"] < 0.3, "EQUIPOISE", "AVOID")))
    print("\n  Subgroup summary (sorted by expected benefit):")
    print(summary.to_string(index=False))
    summary.to_csv(f"{TABLE_DIR}/subgroups.csv", index=False)

    # tree visualisation
    sub("Figure 11a — tree of treatment-effect subgroups")
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(tree, feature_names=feat, filled=True, rounded=True,
              fontsize=8, impurity=False, ax=ax,
              precision=2)
    ax.set_title("Decision tree on CATE  (depth 3)\n"
                 "leaf colour = expected treatment effect (red=harm, blue=benefit)")
    save_fig(fig, f"{FIG_DIR}/11a_subgroup_tree.png")

    # bar chart of subgroup means
    sub("Figure 11b — bar chart of subgroup mean CATE")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1abc9c" if c < -0.3 else "#3498db" if c < 0
              else "#f39c12" if c < 0.3 else "#e74c3c"
              for c in summary["mean_cate"]]
    labels = [f"Leaf {l}\n(n={n}, top: {d[:18]})"
              for l, n, d in zip(summary["leaf"], summary["n"],
                                 summary["top_diagnosis"])]
    ax.barh(labels, summary["mean_cate"], color=colors, alpha=0.85)
    ax.axvline(0, color="black")
    ax.set_xlabel("mean CATE  (days)   ← benefit | harm →")
    ax.set_title("Subgroups discovered by tree on CATE")
    save_fig(fig, f"{FIG_DIR}/11b_subgroup_bars.png")

    print(f"\n  [saved] subgroups.csv  +  11a_subgroup_tree.png  +  11b_subgroup_bars.png")


if __name__ == "__main__":
    main()
