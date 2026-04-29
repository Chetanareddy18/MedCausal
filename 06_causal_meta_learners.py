"""
STEP 6 — Causal ML meta-learners  (S / T / X).

Estimates ATE and per-patient CATE on the real 500 patients.
Persists CATE vectors for the decision system in step 07.
"""
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from causalml.inference.meta import (LRSRegressor, XGBTRegressor,
                                     BaseXRegressor)

from config  import ART, TABLE_DIR, RANDOM_STATE, XGB_PARAMS
from utils   import banner, sub
import importlib

# reuse design-matrix builder
build = importlib.import_module("05_propensity_naive_bayes").build_design_matrix


def safe_ate(triple):
    return tuple(float(np.atleast_1d(t)[0]) for t in triple)


def main() -> None:
    banner("STEP 6 — CAUSAL META-LEARNERS  (S / T / X)")
    df = pd.read_parquet(ART["causal_frame"])
    X, feat = build(df)
    T = df["treatment"].values.astype(int)
    Y = df["outcome"].values.astype(float)
    true_cate = df["true_cate"].values.astype(float)
    true_ate  = float(true_cate.mean())
    T_str = np.where(T == 1, "treatment", "control")

    print(f"  X shape: {X.shape}    treated: {T.sum()}  control: {(T==0).sum()}")
    print(f"  Ground-truth ATE  : {true_ate:+.3f} d")

    learners = {
        "S-Learner (Linear)":
            LRSRegressor(control_name="control"),
        "T-Learner (XGBoost)":
            XGBTRegressor(control_name="control",
                          random_state=RANDOM_STATE),
        "X-Learner (XGBoost)":
            BaseXRegressor(learner=XGBRegressor(**XGB_PARAMS),
                           control_name="control"),
    }

    results = {}
    for name, m in learners.items():
        sub(name)
        cate = m.fit_predict(X=X, treatment=T_str, y=Y).flatten()
        ate, lo, hi = safe_ate(m.estimate_ate(X=X, treatment=T_str, y=Y))
        # ----- ground-truth validation -----
        rmse  = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
        corr  = float(np.corrcoef(cate, true_cate)[0, 1]) if cate.std() > 0 else 0.0
        bias  = float(ate - true_ate)
        print(f"  ATE = {ate:+.3f} d   95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"  CATE — mean {cate.mean():+.3f}  std {cate.std():.3f}  "
              f"benefit% {(cate < 0).mean()*100:.1f}")
        print(f"  vs TRUTH:  ATE bias = {bias:+.3f}  |  "
              f"CATE RMSE = {rmse:.3f}  |  Pearson r = {corr:+.3f}")
        results[name] = {"ate": ate, "lo": lo, "hi": hi, "cate": cate,
                         "ate_bias": bias, "cate_rmse": rmse,
                         "cate_corr": corr}

    # ----- persist
    with open(ART["cate_results"], "wb") as fh:
        pickle.dump({"results": results,
                     "feature_names": feat,
                     "X": X, "T": T, "Y": Y}, fh)

    # summary table
    rows = [{"model": k, "ate": v["ate"], "ci_lo": v["lo"], "ci_hi": v["hi"],
             "cate_mean": v["cate"].mean(),
             "cate_std":  v["cate"].std(),
             "pct_benefit": float((v["cate"] < 0).mean()),
             "ate_bias":   v["ate_bias"],
             "cate_rmse":  v["cate_rmse"],
             "cate_corr":  v["cate_corr"]}
            for k, v in results.items()]
    pd.DataFrame(rows).to_csv(f"{TABLE_DIR}/causal_ate_summary.csv",
                              index=False)
    print(f"\n  [saved] {ART['cate_results']}")


if __name__ == "__main__":
    main()
