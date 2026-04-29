"""
STEP 12 — Sensitivity analysis  (E-value + simulated unmeasured confounding).

Question a skeptical doctor will ask:
   "What if there's some hidden confounder you didn't measure
    (e.g. doctor's gut feeling about the patient)?  How much would
    that hidden variable have to bias things to wipe out your effect?"

We answer with two complementary tools:

  1. E-value (VanderWeele & Ding, 2017) — minimum strength of association
     (on the risk-ratio scale) that an unmeasured confounder would need
     with BOTH treatment and outcome to reduce the observed effect to null.
     Higher E-value ⇒ more robust finding.

  2. Simulation: inject an unmeasured confounder U with controlled
     strength (γ_T affects T, γ_Y affects Y) and re-estimate the IPW-ATE.
     We sweep γ and plot the bias surface.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from config import ART, FIG_DIR, TABLE_DIR, RANDOM_STATE
from utils  import banner, sub, save_fig
import importlib

build = importlib.import_module("05_propensity_naive_bayes").build_design_matrix


def evalue_continuous(estimate_days: float, sd_outcome: float) -> float:
    """E-value for a continuous outcome on the standardised mean-difference
    scale.  Convert to risk-ratio approximation (VanderWeele 2020)."""
    if estimate_days == 0 or np.isnan(estimate_days):
        return 1.0
    d = abs(estimate_days) / sd_outcome           # standardised effect
    rr = float(np.exp(0.91 * d))                  # Chinn (2000) approximation
    return float(rr + np.sqrt(rr * (rr - 1)))


def ipw_ate(t, y, p):
    p = np.clip(p, 0.05, 0.95)
    return ((t * y / p).sum() / (t / p).sum()
            - ((1 - t) * y / (1 - p)).sum() / ((1 - t) / (1 - p)).sum())


def main() -> None:
    banner("STEP 12 — SENSITIVITY ANALYSIS  (E-value  +  hidden confounder sweep)")
    df = pd.read_parquet(ART["causal_frame"])
    X, _ = build(df)
    T = df["treatment"].values.astype(int)
    Y = df["outcome"].values.astype(float)

    naive = Y[T == 1].mean() - Y[T == 0].mean()
    nb    = GaussianNB().fit(X, T)
    ps    = nb.predict_proba(X)[:, 1]
    ipw   = ipw_ate(T, Y, ps)
    sd_y  = Y.std()
    print(f"  Naive ATE   : {naive:+.3f} d")
    print(f"  IPW   ATE   : {ipw:+.3f} d")
    print(f"  SD outcome  : {sd_y:.3f} d")

    # ------------------------------------------------------- 12.1 E-value
    sub("12.1  E-value")
    e_naive = evalue_continuous(naive, sd_y)
    e_ipw   = evalue_continuous(ipw,   sd_y)
    print(f"  E-value (Naive ATE)        : {e_naive:.2f}")
    print(f"  E-value (IPW-NB ATE)       : {e_ipw:.2f}")
    print(f"  Interpretation: an unmeasured confounder would need")
    print(f"  RR ≥ {e_ipw:.2f} with BOTH treatment and outcome to fully")
    print(f"  explain away the IPW-ATE finding.")

    # ------------------------------------------------------- 12.2 sim sweep
    sub("12.2  Hidden-confounder simulation sweep")
    rng = np.random.default_rng(RANDOM_STATE)
    U   = rng.normal(0, 1, len(Y))                 # latent confounder

    grid = np.linspace(0.0, 2.0, 11)               # 0 .. strong
    bias_surface = np.zeros((len(grid), len(grid)))
    for i, gT in enumerate(grid):
        # build a NEW propensity that "knows" U but the analyst doesn't
        logits = nb.predict_log_proba(X)[:, 1] + gT * U
        p_true = 1 / (1 + np.exp(-logits))
        for j, gY in enumerate(grid):
            y_adj = Y + gY * U                      # outcome shifted by U
            ate_obs = ipw_ate(T, y_adj, ps)         # analyst still uses the
                                                    # U-blind propensity
            bias_surface[i, j] = ate_obs - ipw      # extra bias vs no-U case

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(bias_surface, origin="lower",
                   extent=(grid.min(), grid.max(), grid.min(), grid.max()),
                   aspect="auto", cmap="RdBu_r",
                   vmin=-abs(bias_surface).max(),
                   vmax=+abs(bias_surface).max())
    cs = ax.contour(grid, grid, bias_surface,
                    levels=[-ipw], colors="black", linewidths=2)
    ax.clabel(cs, fmt={cs.levels[0]: "FLIP zone"}, fontsize=10)
    ax.set_xlabel("γ_Y  (strength of U → outcome)")
    ax.set_ylabel("γ_T  (strength of U → treatment)")
    ax.set_title(f"Bias from a hidden confounder U\n"
                 f"black contour = strength needed to wipe out IPW-ATE "
                 f"({ipw:+.2f} d)")
    plt.colorbar(im, ax=ax, label="extra bias on IPW-ATE (days)")
    save_fig(fig, f"{FIG_DIR}/12a_sensitivity_surface.png")

    pd.DataFrame({
        "metric": ["naive_ate", "ipw_ate", "sd_outcome",
                   "evalue_naive", "evalue_ipw"],
        "value":  [naive, ipw, sd_y, e_naive, e_ipw],
    }).to_csv(f"{TABLE_DIR}/sensitivity.csv", index=False)
    print(f"  [saved] sensitivity.csv  +  12a_sensitivity_surface.png")


if __name__ == "__main__":
    main()
