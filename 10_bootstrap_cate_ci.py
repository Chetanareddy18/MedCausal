"""
STEP 10 — Per-patient CATE confidence intervals via bootstrap.

WHY: A point estimate like "this patient benefits 1.4 days" is dishonest
if we don't show uncertainty.  Doctors must SEE confidence intervals
or they will (correctly) distrust the model.

We resample the dataset B times, refit the X-Learner, and store the
resulting CATE distribution per patient.  Output: 95% CI per patient.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from causalml.inference.meta import BaseXRegressor

from config import ART, FIG_DIR, TABLE_DIR, RANDOM_STATE, XGB_PARAMS
from utils  import banner, sub, save_fig

N_BOOT = 200          # 200 X-Learner refits — heavy but tractable on 500 pts


def main() -> None:
    banner("STEP 10 — BOOTSTRAP CATE CONFIDENCE INTERVALS")
    with open(ART["cate_results"], "rb") as fh:
        bundle = pickle.load(fh)
    X = bundle["X"]; T = bundle["T"]; Y = bundle["Y"]
    n = len(Y)

    rng = np.random.default_rng(RANDOM_STATE)
    T_str = np.where(T == 1, "treatment", "control")

    sub(f"Refitting X-Learner on {N_BOOT} bootstrap samples (n={n})")
    cate_boot = np.zeros((N_BOOT, n))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = BaseXRegressor(learner=XGBRegressor(**XGB_PARAMS),
                           control_name="control")
        m.fit(X=X[idx], treatment=T_str[idx], y=Y[idx])
        cate_boot[b] = m.predict(X=X).flatten()
        if (b + 1) % 25 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}")

    sub("Per-patient summaries")
    cate_lo  = np.percentile(cate_boot,  2.5, axis=0)
    cate_hi  = np.percentile(cate_boot, 97.5, axis=0)
    cate_med = np.percentile(cate_boot, 50.0, axis=0)
    cate_se  = cate_boot.std(axis=0)

    df = pd.read_parquet(ART["causal_frame"])
    out = pd.DataFrame({
        "patient_id":  df["patient_id"].values,
        "cate_median": cate_med,
        "cate_lo95":   cate_lo,
        "cate_hi95":   cate_hi,
        "cate_se":     cate_se,
        "true_cate":   df["true_cate"].values,
    })
    # CI properties
    width = (cate_hi - cate_lo)
    sig_benefit = (cate_hi < 0).mean() * 100   # CI fully below 0
    sig_harm    = (cate_lo > 0).mean() * 100   # CI fully above 0
    coverage    = ((cate_lo <= df["true_cate"].values) &
                   (df["true_cate"].values <= cate_hi)).mean() * 100
    print(f"  Median CI half-width   : {width.mean()/2:.2f} d")
    print(f"  Patients w/ CI<0       : {sig_benefit:5.1f}%   (significant benefit)")
    print(f"  Patients w/ CI>0       : {sig_harm:5.1f}%   (significant harm)")
    print(f"  Truth coverage of 95%CI: {coverage:5.1f}%   (target ≈ 95%)")

    out.to_csv(f"{TABLE_DIR}/cate_with_ci.csv", index=False)

    sub("Figure 10a — uncertainty caterpillar (50 random patients)")
    sample = out.sort_values("cate_median").reset_index(drop=True)
    sample = sample.iloc[np.linspace(0, len(sample)-1, 50).astype(int)]
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(sample))
    ax.errorbar(sample["cate_median"], y_pos,
                xerr=[sample["cate_median"]-sample["cate_lo95"],
                      sample["cate_hi95"]-sample["cate_median"]],
                fmt="o", color="#2c3e50", ecolor="#7f8c8d",
                elinewidth=1.2, capsize=2, markersize=4,
                label="X-Learner CATE  (95% CI)")
    ax.scatter(sample["true_cate"], y_pos, marker="x",
               color="#e67e22", s=40, label="ground truth τ")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks([])
    ax.set_xlabel("Treatment effect (days)   negative = treatment shortens stay")
    ax.set_title("Per-patient CATE with bootstrap 95% CI")
    ax.legend(loc="lower right")
    save_fig(fig, f"{FIG_DIR}/10a_cate_caterpillar.png")

    # save raw bootstrap matrix for the dashboard
    np.save(f"{ART['cate_results'].replace('cate_results.pkl', 'cate_boot.npy')}",
            cate_boot)
    print(f"  [saved] cate_with_ci.csv   +   cate_boot.npy")


if __name__ == "__main__":
    main()
