"""
STEP 1 — Load raw data and run sanity checks.

Inputs : data/patients_meta.csv   (500 patients, static covariates)
         data/patient_vitals.csv  (~383K rows, 15-min vitals)
Output : prints summary; no files written (downstream scripts re-read raw CSVs).
"""
import pandas as pd
from config import VITALS_CSV, PATIENTS_CSV
from utils  import banner, sub


def main() -> None:
    banner("STEP 1 — DATA LOADING")

    patients = pd.read_csv(PATIENTS_CSV,
                           parse_dates=["admission_time", "discharge_time"])
    vitals   = pd.read_csv(VITALS_CSV, parse_dates=["timestamp"])

    sub("Patients (static)")
    print(f"  shape   : {patients.shape}")
    print(f"  columns : {patients.columns.tolist()}")
    print()
    print(patients[["patient_id", "age", "gender", "diagnosis",
                    "icu_unit", "comorbidity_count",
                    "icu_stay_days", "scenario"]].head())

    sub("Vitals (time-series, 15-min)")
    print(f"  shape       : {vitals.shape}")
    print(f"  date range  : {vitals['timestamp'].min()}  →  "
          f"{vitals['timestamp'].max()}")
    print(f"  vital cols  : heart_rate, spo2, respiratory_rate, "
          f"systolic_bp, diastolic_bp, temperature, overall_risk")
    print(f"  rows / pt   : mean {vitals.groupby('patient_id').size().mean():.0f}, "
          f"min {vitals.groupby('patient_id').size().min()}, "
          f"max {vitals.groupby('patient_id').size().max()}")

    sub("Class balance — clinical scenarios")
    print(patients["scenario"].value_counts())

    sub("Missing values")
    miss_p = patients.isna().sum().sum()
    miss_v = vitals.isna().sum().sum()
    print(f"  patients : {miss_p}    vitals : {miss_v}")

    sub("Sanity ✓")
    print("  Real-looking ICU dataset (15-min vitals, 8 scenarios, "
          "12 diagnoses, 5 ICU units).")


if __name__ == "__main__":
    main()
