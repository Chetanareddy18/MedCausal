"""
STEP 3 — Define treatment T and outcome Y, with a clinically realistic
         data-generating process (DGP) overlay so causal models have a
         known ground-truth to recover.

================================================================
WHY A DGP OVERLAY?
================================================================
The 500-patient simulator produced clinically realistic VITALS but the
`icu_stay_days` field it shipped is essentially random (Pearson r ≈ 0.05
with every covariate). With no real causal signal there is nothing for
causal ML to estimate.

We therefore overlay a transparent DGP on the outcome:

    base_y(x)  = 4
                 + 2.5 * overall_risk_mean(x)
                 + 0.04 * (age − 50)
                 + 0.30 * comorbidity_count
                 + 0.05 * instability_index
                 + 1.5  * pct_critical
    τ(x)       = -2.0  if  overall_risk_mean ≥ 1.5    (high-risk: big benefit)
                 -0.6  if  1.0 ≤ overall_risk_mean < 1.5  (moderate benefit)
                 +0.4  if  overall_risk_mean < 1.0    (low-risk: over-treatment harm)
    Y_obs      = clip(base_y(x) + τ(x) * T + N(0, 0.6²),  1, 30)

Treatment assignment follows the original scenario flag (T = 1 for
fever_infection / hypoxic_deterioration / hemodynamic_instability /
septic_shock).  Because sicker patients are MORE likely to be in those
scenarios, T is correlated with severity → realistic confounding.

Ground-truth columns saved alongside Y so we can score the meta-learners
against the truth in step 06 / 09.
================================================================
"""
import numpy as np
import pandas as pd

from config import ART, TREATMENT_SCENARIOS, RANDOM_STATE
from utils  import banner, sub


def main() -> None:
    banner("STEP 3 — TREATMENT, OUTCOME & GROUND-TRUTH DGP")
    rng = np.random.default_rng(RANDOM_STATE)
    df  = pd.read_parquet(ART["features"])

    # ---------- treatment (real, scenario-driven, confounded by severity)
    df["treatment"] = df["scenario"].isin(TREATMENT_SCENARIOS).astype(int)

    # ---------- baseline outcome (no treatment effect yet) -----
    risk = df["overall_risk_mean"].values
    base_y = (
        4.0
        + 2.5 * risk
        + 0.04 * (df["age"].values - 50)
        + 0.30 * df["comorbidity_count"].values
        + 0.05 * df["instability_index"].values
        + 1.5  * df["pct_critical"].values
    )

    # ---------- heterogeneous true treatment effect tau(x) -----
    tau = np.where(risk >= 1.5, -2.0,
          np.where(risk >= 1.0, -0.6, +0.4))

    noise = rng.normal(0, 0.6, size=len(df))
    y_obs = np.clip(base_y + tau * df["treatment"].values + noise, 1, 30)

    df["outcome"]            = y_obs        # what models see
    df["true_cate"]          = tau          # ground truth (hidden from models)
    df["base_y"]             = base_y       # for diagnostics
    df["icu_stay_days_raw"]  = df["icu_stay_days"]
    df["icu_stay_days"]      = y_obs        # overlay

    sub("Treatment assignment (real scenario-driven)")
    print(f"  T = 1 (active intervention) : "
          f"{int(df['treatment'].sum())} patients")
    print(f"  T = 0 (monitoring/recovery) : "
          f"{int((df['treatment'] == 0).sum())} patients")
    print(f"  Active scenarios            : {sorted(TREATMENT_SCENARIOS)}")

    sub("Outcome statistics")
    print(df.groupby("treatment")["outcome"]
            .agg(["mean", "std", "median", "min", "max", "count"]).round(2))

    sub("Severity by treatment (confounding evidence)")
    print(df.groupby("treatment")["overall_risk_mean"]
            .agg(["mean", "std"]).round(3))
    print("  → treated patients are sicker; pure observational comparison "
          "will be biased upward.")

    sub("Ground-truth treatment effect distribution (τ)")
    print(pd.Series(tau).value_counts().sort_index().rename("count"))
    true_ate = tau.mean()
    print(f"\n  TRUE ATE  (E[τ])     : {true_ate:+.3f} days")
    print(f"  TRUE pct benefiting  : {(tau < 0).mean()*100:.1f} %")

    df.to_parquet(ART["causal_frame"], index=False)
    print(f"\n  [saved] {ART['causal_frame']}")


if __name__ == "__main__":
    main()
