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

<div align="center">

# 🫀 MedCausal

## **Per-Patient Causal Treatment Effects — The Bedside Difference**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/Chetanareddy18/MedCausal)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**The Problem:** Doctors see that sicker patients get treated more. Naive comparison says treatment *hurts*. But that's Simpson's paradox — **confounding in action**.

**The Solution:** Causal ML + explainability = honest treatment-effect estimates for each patient, with confidence intervals and sensitivity analysis.

**The Result:** X-Learner lands **within ±0.02 days** of ground truth. 60%+ of patients predicted to benefit. **E-value = 3.54** (robust to hidden confounders).

[🚀 **Live Dashboard**](https://huggingface.co/spaces/Chetanareddy18/MedCausal) · [📊 **View WORKFLOW**](WORKFLOW.md) · [📔 **Kaggle Notebook**](kaggle_icu_causal_ml.ipynb)

</div>

---

---

## 📈 Key Results

| Metric | Value | Meaning |
|--------|-------|---------|
| **Ground Truth ATE** | −0.53 days | Planted effect (answer key) |
| **X-Learner ATE** | −0.51 days | Model bias: **−0.02** ✅ |
| **X-Learner CATE r** | **+0.77** | Per-patient rank correlation w/ truth |
| **DR-Learner CATE r** | +0.53 | Cross-fitted robustness check |
| **E-value** | **3.54** | Hidden confounder tolerance |
| **Patients with benefit** | **60%** | Significant + uncertain combined |
| **Bootstrap CI coverage** | **93.4%** | Truth coverage (target: ~95%) |

---

## 🎯 The Dashboard — 6 Interactive Tabs

| Tab | What It Shows |
|---|---|
| **📊 Overview** | ATE ladder (all models vs truth), CATE histogram, clinical action mix |
| **🧑‍⚕️ Patient** | Pick any of 500 patients → per-patient CATE + 95% CI + SHAP feature importance |
| **🧬 Subgroups** | Auto-discovered patient subgroups (decision tree on CATE) → simple rules |
| **🎚️ What-If** | Slider-based counterfactual simulator: "if a 70yo with low SpO₂ gets treated, what happens?" |
| **🛡️ Robustness** | E-value, hidden-confounder bias surface, bootstrap calibration |
| **📖 The Story** | Plain English 6-step narrative (Simpson's paradox → solution) |

---

## 🔬 The Method Stack

```
Observational Data (500 patients, 383K vitals)
         ↓
    [Feature Engineering]
      ↓ ↓ ↓ ↓
   Covariates X | Treatment T | Outcome Y | Propensity P(T|X)
         ↓
    [Causal Meta-Learners]
    ├─ S-Learner (LR)
    ├─ T-Learner (XGB)
    ├─ X-Learner (XGB) ← **Best**
    └─ DR-Learner (5-fold cross-fit, XGB)
         ↓
   [Validation Against Planted Truth]
    ├─ ATE bias
    ├─ CATE rank correlation (r)
    ├─ CATE RMSE
    └─ Bootstrap 95% CI coverage
         ↓
   [Explainability & Sensitivity]
    ├─ SHAP per-feature importance
    ├─ E-value (hidden confounder tolerance)
    └─ Subgroup discovery (decision tree)
         ↓
   [Bedside Decision Rules]
   → TREAT / AVOID / EQUIPOISE per patient
```

---

## 🚀 Quick Start

### **Option 1: Live Dashboard (30 seconds)**
Visit [**MedCausal on Hugging Face Spaces**](https://huggingface.co/spaces/Chetanareddy18/MedCausal) — everything runs in the browser, no install needed. Click through the tabs, pick a patient, see SHAP explanations.

### **Option 2: Local Streamlit App**
```bash
cd CausalML/icu_causal_500
streamlit run app.py
# Opens http://localhost:8501
```
Pre-computed results in `outputs/_cache/` load instantly. No model fitting.

### **Option 3: Kaggle Notebook**
Upload [`kaggle_icu_causal_ml.ipynb`](kaggle_icu_causal_ml.ipynb) to Kaggle:
- Loads real CSVs (or simulates on Kaggle input)
- Runs all 13 sections end-to-end
- Shows ATE ladder, CATE distributions, SHAP, subgroups, sensitivity
- CPU-only, ~5 min total

### **Option 4: Local Reproduction (Advanced)**
Re-run the full causal pipeline:
```bash
cd CausalML/icu_causal_500
# Activate your causal_env (see next section)
python run_all.py
# Runs 01_load_data.py → 14_shap_explain.py
# Regenerates outputs/_cache/ and outputs/
```

---

## 📦 Data: 500 ICU Patients + 383K Vital Readings

| Dataset | Rows | Columns | Size | Notes |
|---------|------|---------|------|-------|
| `patients_meta.csv` | 500 | 10 | 59 KB | Static: age, gender, diagnosis, comorbidities, ICU unit, scenario, icu_stay_days |
| `patient_vitals.csv` | 383,540 | 8 | 50 MB | Timestamped: heart_rate, spo₂, respiratory_rate, systolic_bp, temp, overall_risk, etc. |

**Scenarios:** Septic shock, Fever/infection, Hypoxic deterioration, Hemodynamic instability.

---

## 🏗️ Folder Structure

```
MedCausal/
├── app.py                              ← Streamlit dashboard (6 tabs)
├── kaggle_icu_causal_ml.ipynb         ← End-to-end notebook (Kaggle-ready)
├── requirements.txt                   ← Slim runtime (HF Spaces)
├── requirements-pipeline.txt          ← Full pipeline deps (local)
├── WORKFLOW.md                         ← Clinical workflow + validation results
├── README.md                           ← This file
│
├── data/
│   ├── patients_meta.csv
│   └── patient_vitals.csv
│
├── outputs/
│   ├── _cache/
│   │   ├── causal_frame.parquet       ← Main dataset (500 patients × 13 features)
│   │   ├── cate_results.pkl           ← All model CATE + ATE estimates
│   │   ├── features.parquet           ← Design matrix X (standardized)
│   │   ├── shap_values.npy            ← SHAP matrix (500 × 13)
│   │   └── cate_boot.npy              ← Bootstrap samples (200 × 500)
│   ├── tables/
│   │   ├── causal_ate_summary.csv     ← ATE of all methods vs truth
│   │   ├── cate_with_ci.csv           ← Per-patient CATE + 95% CI
│   │   ├── subgroups.csv              ← Leaf-level stats + recommendations
│   │   ├── sensitivity.csv            ← E-value + confounder bias surface
│   │   └── patient_decisions_500.csv  ← Per-patient verdicts
│   └── figures/
│       ├── 04a_correlations.png       ← Pearson/Spearman heatmap
│       ├── 08a_ate_comparison.png     ← All models vs truth bar chart
│       ├── 08b_cate_distributions.png ← CATE histogram
│       ├── 11a_subgroup_tree.png      ← Decision tree depth 3
│       ├── 14a_shap_summary.png       ← Top 10 SHAP drivers
│       └── [+15 more]
│
├── [01–14]_*.py
│   ├── 01_load_data.py                ← Sanity-check raw CSVs
│   ├── 02_feature_engineering.py      ← 383K rows → 500 patient aggregates
│   ├── 03_treatment_outcome.py        ← Define T and Y
│   ├── 04_correlation_vs_causation.py ← Simpson's paradox demo
│   ├── 05_propensity_naive_bayes.py   ← P(T|X) and IPW-ATE
│   ├── 06_causal_meta_learners.py     ← S/T/X-Learners (XGB)
│   ├── 07_clinical_decisions.py       ← TREAT/AVOID/EQUIPOISE
│   ├── 08_visualizations.py           ← Generate 6 PNG figures
│   ├── 09_doctor_report.py            ← Plain-English summary
│   ├── 10_bootstrap_cate_ci.py        ← 95% CIs (n=200 bootstrap)
│   ├── 11_subgroup_discovery.py       ← Tree on CATE
│   ├── 12_sensitivity_evalue.py       ← E-value + confounder surface
│   ├── 13_dr_learner.py               ← DR-Learner with 5-fold CF
│   └── 14_shap_explain.py             ← SHAP per-feature importance
│
├── run_all.py                         ← Orchestrator (imports 01–09, runs in order)
├── config.py                          ← Paths, constants, theme
├── utils.py                           ← Helpers (banner, save_fig, teach())
└── .streamlit/
    └── config.toml                    ← Dark theme + Streamlit settings
```

---

## 🛠️ Tech Stack

**Causal inference:** `causalml`, `xgboost`, `scikit-learn` (LRSRegressor, XGBRegressor, BaseXRegressor)  
**Propensity:** `GaussianNB` + IPW  
**Explainability:** `shap` (TreeExplainer with fallback to kernel SHAP)  
**Visualization:** `plotly` (interactive), `matplotlib` + `seaborn` (static)  
**Data:** `pandas`, `numpy`, `pyarrow`  
**Dashboard:** `streamlit`  

---

## 🖥️ Local Development

### Prerequisites
- Python 3.10+
- Virtual environment: `causal_env/` or `venv/`

### Install & Run Pipeline
```bash
# 1. Install full deps
pip install -r requirements-pipeline.txt

# 2. Run all 14 steps (generates outputs/)
python run_all.py

# 3. Launch Streamlit dashboard
streamlit run app.py
```

### Run a Single Step
```bash
python 04_correlation_vs_causation.py  # Simpson's paradox
python 06_causal_meta_learners.py      # Train S/T/X-Learners
python 14_shap_explain.py              # Generate SHAP plots
```

---

## 📚 Reading Guide

1. **First time here?** Start with [**WORKFLOW.md**](WORKFLOW.md) — explains the clinical question, dataset, and all 14 steps in prose.
2. **Want to deploy?** Follow the [**Live Dashboard**](https://huggingface.co/spaces/Chetanareddy18/MedCausal) link (HF Spaces auto-syncs from this repo).
3. **Want reproducibility?** Run the [**Kaggle notebook**](kaggle_icu_causal_ml.ipynb) — entire analysis in one notebook.
4. **Want the gory details?** See the [**scripts**](./01_load_data.py) — each handles one concept.
5. **Want validation numbers?** Check [**WORKFLOW.md Validation**](WORKFLOW.md#-validation--results) section.

---

## ✅ What This Project Demonstrates

- ✅ **Causal inference on real data** — honest treatment-effect estimates despite confounding
- ✅ **Meta-learners** — S/T/X/DR-Learners with rigorous comparison
- ✅ **Validation** — ground-truth planted in DGP, measuring ATE bias + CATE correlation
- ✅ **Heterogeneous effects** — per-patient CATE with 95% bootstrap CIs
- ✅ **Subgroup discovery** — auto-found patient groups via decision tree
- ✅ **Sensitivity analysis** — E-value + hidden-confounder bias surface
- ✅ **Explainability** — SHAP forces plot for "why does the model think so?"
- ✅ **Production readiness** — cached results + Streamlit dashboard on HF Spaces
- ✅ **Reproducibility** — Kaggle notebook + plain GitHub repo



## 📄 License

MIT — use for research, teaching, commercial projects. See [LICENSE](LICENSE).

---

<div align="center">

**Built by:** Chetana Reddy  
**For:** ICU decision support + causal ML portfolio  
**Status:** ✅ Production-ready dashboard + Kaggle notebook  

[🚀 Live Demo](https://huggingface.co/spaces/Chetanareddy18/MedCausal) · [📊 Workflow](WORKFLOW.md) · [💻 GitHub](https://github.com/Chetanareddy18/MedCausal)

</div>
