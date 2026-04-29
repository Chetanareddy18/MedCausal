"""
STEP 13 — Doubly-Robust learner with cross-fitting.

A DR-Learner combines:
  • outcome regression  μ̂(t, x)
  • propensity model    ê(x)
into a "pseudo-outcome" φ̂_i that is consistent if EITHER the outcome
model OR the propensity model is correct.  We then regress φ̂ on X.

Cross-fitting (K-fold sample splitting, Chernozhukov et al. 2018) avoids
own-observation bias and is the recipe behind modern, publishable causal
ML.  This is the most rigorous estimator in the project.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from config import ART, FIG_DIR, TABLE_DIR, RANDOM_STATE, XGB_PARAMS
from utils  import banner, sub, save_fig
import importlib

build = importlib.import_module("05_propensity_naive_bayes").build_design_matrix


def main() -> None:
    banner("STEP 13 — DR-LEARNER  (doubly-robust + 5-fold cross-fitting)")
    df = pd.read_parquet(ART["causal_frame"])
    X, feat = build(df)
    T = df["treatment"].values.astype(int)
    Y = df["outcome"].values.astype(float)
    true_cate = df["true_cate"].values.astype(float)
    n = len(Y)

    sub("5-fold cross-fitted nuisance models")
    mu0 = np.zeros(n); mu1 = np.zeros(n); ps = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        # outcome models on training fold
        m0 = XGBRegressor(**XGB_PARAMS)
        m0.fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0])
        m1 = XGBRegressor(**XGB_PARAMS)
        m1.fit(X[tr][T[tr] == 1], Y[tr][T[tr] == 1])
        # propensity on training fold
        e  = GaussianNB().fit(X[tr], T[tr])
        # predict on held-out fold (no leakage)
        mu0[te] = m0.predict(X[te])
        mu1[te] = m1.predict(X[te])
        ps[te]  = np.clip(e.predict_proba(X[te])[:, 1], 0.05, 0.95)
        print(f"  fold {fold}/5 done")

    sub("Pseudo-outcome φ̂  +  cross-fitted CATE regression")
    # AIPW pseudo-outcome (clip the residual term to tame heavy tails)
    resid_t = np.clip(T * (Y - mu1) / ps,             -10, 10)
    resid_c = np.clip((1 - T) * (Y - mu0) / (1 - ps), -10, 10)
    phi = (mu1 - mu0) + resid_t - resid_c

    # second-stage learner on φ̂ — also cross-fit so per-patient CATE
    # is not predicted on the data its model was trained on
    cate = np.zeros(n)
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr], phi[tr])
        cate[te] = m.predict(X[te])
    ate  = float(phi.mean())            # IF-based ATE (more efficient)

    # influence-function variance for ATE
    se   = float(phi.std() / np.sqrt(n))
    lo, hi = ate - 1.96 * se, ate + 1.96 * se

    # validation
    rmse = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
    corr = float(np.corrcoef(cate, true_cate)[0, 1])
    bias = ate - true_cate.mean()
    print(f"  DR-Learner ATE        = {ate:+.3f} d   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  vs TRUTH:  bias = {bias:+.3f}  |  CATE RMSE = {rmse:.3f}  "
          f"|  Pearson r = {corr:+.3f}")

    # save into the cate_results bundle for the dashboard
    with open(ART["cate_results"], "rb") as fh:
        bundle = pickle.load(fh)
    bundle["results"]["DR-Learner (XGB, 5-fold)"] = {
        "ate": ate, "lo": lo, "hi": hi, "cate": cate,
        "ate_bias": bias, "cate_rmse": rmse, "cate_corr": corr,
    }
    with open(ART["cate_results"], "wb") as fh:
        pickle.dump(bundle, fh)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(true_cate, cate, alpha=0.5, s=22, color="#16a085")
    lo_, hi_ = float(min(true_cate.min(), cate.min())), \
               float(max(true_cate.max(), cate.max()))
    ax[0].plot([lo_, hi_], [lo_, hi_], "r--", lw=1, label="perfect recovery")
    ax[0].set_xlabel("TRUE τ")
    ax[0].set_ylabel("DR-Learner CATE")
    ax[0].set_title(f"DR-Learner recovery   r={corr:+.2f}   RMSE={rmse:.2f}")
    ax[0].legend()
    ax[1].hist(cate, bins=30, color="#16a085", alpha=0.85)
    ax[1].axvline(ate, color="black",  lw=2, label=f"ATE {ate:+.2f}")
    ax[1].axvline(0,    color="grey",  lw=1, ls="--")
    ax[1].set_xlabel("CATE (days)")
    ax[1].set_title("DR-Learner CATE distribution")
    ax[1].legend()
    save_fig(fig, f"{FIG_DIR}/13a_dr_learner.png")

    print(f"  [saved] 13a_dr_learner.png  (DR-Learner appended to cate_results)")


if __name__ == "__main__":
    main()
