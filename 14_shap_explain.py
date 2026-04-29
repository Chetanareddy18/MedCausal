"""
STEP 14 — SHAP explainability of the X-Learner CATE.

Doctors will ask: "WHY does the model think this patient benefits?"
We answer with SHAP values computed on a surrogate XGBoost regressor
fitted to the X-Learner CATE.  Two outputs:
  • global feature importance (which drivers matter for heterogeneity)
  • a saved SHAP matrix per patient → consumed by the Streamlit app
"""
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from config import ART, FIG_DIR, TABLE_DIR, XGB_PARAMS
from utils  import banner, sub, save_fig

try:
    import shap
except ImportError as e:
    raise SystemExit("shap not installed.  pip install shap") from e


def main() -> None:
    banner("STEP 14 — SHAP EXPLAINABILITY (per-patient drivers of CATE)")
    with open(ART["cate_results"], "rb") as fh:
        bundle = pickle.load(fh)
    cate = bundle["results"]["X-Learner (XGBoost)"]["cate"]
    feat = bundle["feature_names"]
    X    = bundle["X"]

    sub("Surrogate XGBoost on CATE")
    surrogate = XGBRegressor(**XGB_PARAMS)
    surrogate.fit(X, cate)
    r2 = float(np.corrcoef(surrogate.predict(X), cate)[0, 1] ** 2)
    print(f"  Surrogate R^2 vs CATE = {r2:.3f}")

    sub("Computing SHAP values")
    explainer = shap.TreeExplainer(surrogate)
    sv = explainer.shap_values(X)
    print(f"  SHAP matrix shape = {sv.shape}")

    # global importance
    imp = pd.DataFrame({"feature": feat,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(f"{TABLE_DIR}/shap_global_importance.csv", index=False)
    print("\n  Top-10 drivers of CATE heterogeneity:")
    print(imp.head(10).to_string(index=False))

    # save shap matrix for dashboard
    np.save(ART["cate_results"].replace("cate_results.pkl", "shap_values.npy"), sv)
    np.save(ART["cate_results"].replace("cate_results.pkl", "shap_X.npy"),      X)

    # global summary plot
    sub("Figure 14a — SHAP summary (global)")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X, feature_names=feat, show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle("SHAP — drivers of treatment-effect heterogeneity")
    save_fig(fig, f"{FIG_DIR}/14a_shap_summary.png")

    # bar of top drivers
    sub("Figure 14b — top driver bar")
    fig, ax = plt.subplots(figsize=(9, 6))
    top = imp.head(12).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#8e44ad", alpha=0.85)
    ax.set_xlabel("mean |SHAP value|  (impact on CATE)")
    ax.set_title("Top 12 features driving who benefits from treatment")
    save_fig(fig, f"{FIG_DIR}/14b_shap_top.png")

    print(f"  [saved] shap_global_importance.csv  +  shap_values.npy  "
          f"+  14a/b figures")


if __name__ == "__main__":
    main()
