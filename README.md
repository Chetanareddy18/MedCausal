---
title: MedCausal
emoji: 🫀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.37.0
app_file: app.py
pinned: false
license: mit
short_description: ICU Vitals Causal — per-patient treatment effect dashboard with explainability
---

# MedCausal — ICU Vitals Causal

Per-patient causal treatment-effect estimation from observational ICU data (500 patients, 383K vital readings).
Solves Simpson's paradox using meta-learners (X-Learner, DR-Learner) with 95% CIs, SHAP explainability, and E-value sensitivity.

> **[Live Dashboard](https://huggingface.co/spaces/Chetanareddy18/MedCausal)** — The Streamlit app reads cached results from
> `outputs/_cache/` and `outputs/tables/`, so the Hugging Face Space boots in seconds without refitting models.

## Folder layout
```
icu_causal_500/
├── data/                          ← real 500-patient CSVs (copied from TFT/data)
├── outputs/
│   ├── figures/                   ← all PNGs for the deck
│   ├── tables/                    ← CSV results
│   └── _cache/                    ← intermediate parquet/pickle artefacts
├── config.py                      ← paths, constants, treatment rules
├── utils.py                       ← shared helpers (banner, save_fig)
├── 01_load_data.py
├── 02_feature_engineering.py
├── 03_treatment_outcome.py
├── 04_correlation_vs_causation.py ← the headline doctor demo
├── 05_propensity_naive_bayes.py
├── 06_causal_meta_learners.py
├── 07_clinical_decisions.py
├── 08_visualizations.py
├── 09_doctor_report.py
└── run_all.py                     ← runs 01 → 09 in order
```

## Causal contract
| Symbol | Meaning |
|---|---|
| **X** | age, sex, comorbidities, BMI, ICU unit, admission type, diagnosis, vitals summaries |
| **T** | 1 if scenario ∈ {`fever_infection`, `hypoxic_deterioration`, `hemodynamic_instability`, `septic_shock`} else 0 |
| **Y** | `icu_stay_days` (continuous, primary) ; `ever_critical` (binary, secondary) |

## Run

```powershell
# from CausalML/icu_causal_500
..\causal_env\Scripts\python.exe run_all.py
```

Or run any step in isolation, e.g.:
```powershell
..\causal_env\Scripts\python.exe 04_correlation_vs_causation.py
```

## What each step does
1. **Load** — read raw CSVs, sanity check.
2. **Feature engineering** — collapse 383K rows to 500 patient-level rows.
3. **Treatment / outcome** — define T and Y.
4. **Correlation vs causation** — Pearson/Spearman + Simpson's-paradox demo.
5. **Naive Bayes propensity** — P(T=1|X) → IPW-ATE.
6. **Causal meta-learners** — S / T / X learners → ATE + per-patient CATE (validated against ground-truth τ).
7. **Clinical decisions** — TREAT/AVOID per patient + 4-quadrant action map.
8. **Visualisations** — six figures for the deck.
9. **Doctor report** — printed summary + bundled CSV.
