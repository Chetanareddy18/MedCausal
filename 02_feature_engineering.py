"""
STEP 2 — Aggregate minute-level vitals into per-patient features.

Input  : raw CSVs.
Output : outputs/_cache/features.parquet  (one row per patient).

Design : we summarise each vital with central tendency + variability + extremes,
         plus clinically meaningful derived features (time spent critical,
         deterioration rate, vitals instability index).
"""
import numpy as np
import pandas as pd
from config import VITALS_CSV, PATIENTS_CSV, ART
from utils  import banner, sub


# Per-vital aggregation spec — chosen to mirror what an ICU clinician
# would scan on a chart: average state, variability, worst value.
AGG_SPEC = {
    "heart_rate":       ["mean", "std", "min", "max"],
    "spo2":             ["mean", "std", "min"],
    "respiratory_rate": ["mean", "std", "max"],
    "systolic_bp":      ["mean", "std", "min", "max"],
    "diastolic_bp":     ["mean", "std"],
    "temperature":      ["mean", "max"],
    "overall_risk":     ["mean", "std", "max"],
}


def main() -> None:
    banner("STEP 2 — FEATURE ENGINEERING")

    vitals   = pd.read_csv(VITALS_CSV, parse_dates=["timestamp"])
    patients = pd.read_csv(PATIENTS_CSV)

    sub("Aggregating ~383K rows → 500 patient-level rows")
    agg = vitals.groupby("patient_id").agg(AGG_SPEC)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    # ---------- derived clinical features -----------------------------------
    sub("Deriving clinical risk features")
    derived = vitals.groupby("patient_id").agg(
        n_critical_events = ("overall_risk", lambda s: int((s >= 3).sum())),
        pct_high_risk     = ("overall_risk", lambda s: float((s >= 2).mean())),
        pct_critical      = ("overall_risk", lambda s: float((s >= 3).mean())),
        n_readings        = ("overall_risk", "count"),
        # deterioration trend: slope of risk over the stay (positive = worsening)
        risk_trend        = ("overall_risk",
                             lambda s: np.polyfit(np.arange(len(s)), s, 1)[0]
                                       if len(s) > 5 else 0.0),
    ).reset_index()

    # binary outcome companion
    derived["ever_critical"] = (derived["n_critical_events"] > 0).astype(int)

    # vital-sign instability index — std summed across primary vitals
    instability_components = ["heart_rate_std", "spo2_std",
                              "respiratory_rate_std", "systolic_bp_std"]
    agg["instability_index"] = agg[instability_components].sum(axis=1)

    # merge everything onto static patient table
    out = patients.merge(agg, on="patient_id").merge(derived, on="patient_id")

    sub("Resulting feature frame")
    print(f"  shape   : {out.shape}")
    print(f"  example features:")
    print("   ", [c for c in out.columns
                  if c not in patients.columns][:10], "...")

    out.to_parquet(ART["features"], index=False)
    print(f"\n  [saved] {ART['features']}")


if __name__ == "__main__":
    main()
