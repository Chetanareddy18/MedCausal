# ICU Causal-ML — End-to-End Workflow

> Estimating **per-patient** treatment effects in the ICU using causal machine learning.
> 500 simulated ICU patients · 383,540 vital-sign readings · 14-step validated pipeline.

---

## 1. The clinical question

> *"Does the active intervention bundle (antibiotics + O₂ support + pressors) actually
> shorten ICU stay — and **for which patients**?"*

The trap: in observational data, **sicker patients are treated more often**.
A naive comparison of treated-vs-untreated averages gives the **wrong sign**
(treatment looks harmful) — a textbook **Simpson's paradox / confounding**.

The fix: **causal ML** — estimate Average Treatment Effect (ATE) and per-patient
Conditional Average Treatment Effect (CATE) after adjusting for who-gets-treated.

---

## 2. Dataset

| | |
|--|--|
| Patients | 500 |
| Vital-sign readings | 383,540 |
| Sampling | irregular ICU monitor stream |
| Scenarios | `fever_infection`, `hypoxic_deterioration`, `hemodynamic_instability`, `septic_shock` |
| Treatment | binary — bundle administered or not |
| Outcome | ICU stay (days) |

A **planted ground-truth treatment effect** is overlaid in step 03 so every
model can be graded honestly:

```
true_cate = -2.0d  if overall_risk_mean ≥ 1.5
            -0.6d  if overall_risk_mean ≥ 1.0
            +0.4d  otherwise
```

This lets us measure **bias**, **CATE RMSE**, and **Pearson r vs the truth**.

---

## 3. Pipeline (14 steps)

Each step is a standalone Python file with `main()`. State is cached as
`parquet` / `pickle` in `outputs/_cache/`. `run_all.py` invokes them in order
via `importlib`.

| # | Script | What it does | Why it matters |
|---|--------|--------------|----------------|
| **01** | `01_load_data.py` | Load raw vitals + patient metadata, merge | Single source of truth for every later step |
| **02** | `02_feature_engineering.py` | Per-patient aggregates (mean/std/min/max/pct-critical) | Turns 383K rows into a 500-row tabular frame for ML |
| **03** | `03_treatment_outcome.py` | Plant `true_cate`, `base_y`, treatment assignment, observed outcome | Builds a synthetic but realistic ground-truth so we can grade models |
| **04** | `04_correlation_vs_causation.py` | Pearson, naive ATE, stratified-by-severity ATE | Demonstrates Simpson's paradox — naive comparison flips the sign |
| **05** | `05_propensity_naive_bayes.py` | GaussianNB propensity scores + IPW-ATE | Reweights treated/untreated to look comparable |
| **06** | `06_causal_meta_learners.py` | Fit S-/T-/X-Learner (XGBoost backbone) | Three different ways to estimate CATE; X-Learner is the headliner |
| **07** | `07_clinical_decisions.py` | TREAT / AVOID / EQUIPOISE / MONITOR rule on per-patient CATE | Translates math into bedside actions |
| **08** | `08_visualizations.py` | 7 figures: ATE comparison, CATE distribution, drivers, decision quadrants… | Makes the result legible at a glance |
| **09** | `09_doctor_report.py` | One-page CSV per patient: profile + CATE + verdict | Shareable bedside output |
| **10** | `10_bootstrap_cate_ci.py` | 200 bootstrap X-Learner refits → 95% CI per patient | Honest uncertainty, not pretend-precision |
| **11** | `11_subgroup_discovery.py` | Depth-3 `DecisionTreeRegressor` on CATE → leaves with rules | "Who benefits the most?" — auto-discovered |
| **12** | `12_sensitivity_evalue.py` | E-value (Chinn 2000) + 11×11 hidden-confounder bias surface | Could a hidden factor flip our conclusion? |
| **13** | `13_dr_learner.py` | DR-Learner with 5-fold cross-fitting (Chernozhukov-style) | Doubly-robust ATE with influence-function CI — publishable rigor |
| **14** | `14_shap_explain.py` | Surrogate XGB on CATE → TreeSHAP global + per-patient | "Why does the model think this patient benefits?" |

---

## 4. Validated results

Numbers from the latest end-to-end run.

### Average treatment effect (ATE)

| Method | ATE (days) | Bias vs truth | CATE Pearson r | CATE RMSE |
|--------|-----------:|--------------:|---------------:|----------:|
| **Ground truth (planted)** | **−0.51** | 0.00 | 1.00 | 0.00 |
| Naive (correlation) | **+1.34** | +1.85 (wrong sign!) | — | — |
| S-Learner (Linear) | −0.42 | +0.09 | 0.61 | 1.21 |
| T-Learner (XGBoost) | −0.48 | +0.03 | 0.71 | 1.05 |
| **X-Learner (XGBoost)** | **−0.53** | **−0.02** | **+0.77** | **0.92** |
| DR-Learner (XGB, 5-fold) | −0.27 | +0.24 | +0.53 | 1.43 |
| IPW-NB | −0.38 | +0.13 | — | — |

**Headline:** the X-Learner recovers the ATE to within ±0.02 d of the planted truth
and ranks per-patient effects with r = +0.77 — strong validation that the pipeline
is doing real causal work, not just chasing noise.

### Per-patient uncertainty (bootstrap, n=200)

| Quantity | Value |
|--|--:|
| Patients with CI fully below 0 (significant benefit) | **22.2%** |
| Patients with CI fully above 0 (significant harm) | 4.8% |
| 95% CI coverage of planted truth | **93.4%** (target ≈ 95%) |

### Subgroups (depth-3 tree on CATE, R² = 0.83)

| Leaf | n | Mean CATE | Recommendation | Top diagnosis |
|--|--:|--:|--|--|
| 12 | 42 | **−2.39 d** | **STRONG TREAT** | Septic shock |
| 7  | 38 | −1.62 d | TREAT | Hypoxic deterioration |
| 3  | 71 | −0.41 d | TREAT (uncertain) | Mixed |
| 9  | 28 | +0.92 d | AVOID | Mild COPD |

### Robustness

| Quantity | Value |
|--|--:|
| **E-value (IPW-NB ATE)** | **3.54** |
| E-value (Naive ATE) | 1.21 |
| Sensitivity surface | a hidden confounder needs both γ_T and γ_Y > ~1.7 to flip the IPW result |

### SHAP top drivers (global mean |SHAP| on X-Learner CATE)

1. `overall_risk_mean` — 0.58
2. `instability_index`
3. `pct_critical`
4. `spo2_mean`
5. `comorbidity_count`
6. `age`
7. `respiratory_rate_mean`

---

## 5. Streamlit dashboard (`app.py`)

Six tabs, all backed by the artefacts above:

| Tab | Purpose |
|--|--|
| 📊 Overview | ATE ladder vs truth · CATE histogram · clinical-action mix |
| 🧑‍⚕️ Patient | Per-patient CATE + 95% CI + verdict pill + SHAP top drivers |
| 🧬 Subgroups | Auto-discovered tree, per-leaf cards & bars |
| 🎚️ What-If | Slider-driven counterfactual simulator (surrogate XGB) |
| 🛡️ Robustness | E-value cards · sensitivity surface · bootstrap caterpillar |
| 📖 The Story | Plain-English narrative pulling the whole thing together |

A 💡 collapsed dropdown on every tab teaches the reader what each chart means
and how to read it — built for non-medical, non-statistics readers.

Launch:

```powershell
& "..\causal_env\Scripts\streamlit.exe" run app.py --server.headless true --server.port 8501
```

---

## 6. How to reproduce

```powershell
# 1. Activate the env
& ..\causal_env\Scripts\Activate.ps1

# 2. Install (only first time)
pip install -r requirements.txt

# 3. Run the full pipeline (~7 minutes on a laptop)
python run_all.py

# 4. Launch the dashboard
streamlit run app.py
```

Artefacts land in:

- `outputs/_cache/` — intermediate parquets / pickles
- `outputs/tables/` — CSVs (CIs, subgroups, SHAP importance, sensitivity…)
- `outputs/figures/` — PNGs (subgroup tree, sensitivity surface, caterpillar…)

---

## 7. What I learned (the meta-takeaways)

1. **Correlation actively lies in observational health data.** The naive ATE
   pointed the *wrong direction*. Without the planted truth as ground truth I
   would never have known.
2. **Heterogeneity matters more than the average.** ATE = −0.5 d is a
   one-line summary; the histogram of per-patient CATEs runs from
   −3 d (huge benefit) to +1 d (mild harm). Subgroup tree turns that into
   actionable bedside rules.
3. **Cross-fitting saved the DR-Learner.** A naive single-fit DR run gave
   RMSE 4.79 and r = 0.23 (worse than the simple X-Learner). With proper
   5-fold cross-fitting + clipped IPW residuals, RMSE dropped to 1.43 and
   r climbed to +0.53 — and the CI properly excluded zero.
4. **Robustness is not optional.** The E-value (3.54) is the single number
   I'd quote to a sceptical reviewer: "any hidden confounder would have to
   be 3.5× stronger than anything we measured to wipe this out".
5. **Model + uncertainty + explanation = trust.** The combination of CATE
   point estimates, bootstrap CIs, and SHAP per-patient drivers is what
   turns a black-box prediction into something a clinician would actually
   look at.

---

*Built with `causalml`, `xgboost`, `scikit-learn`, `shap`, `streamlit`, `plotly`.*
