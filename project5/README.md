# Cirrhosis Mortality Risk Prediction — Gradient Boosting + SHAP

Predicting patient mortality risk from Primary Biliary Cirrhosis (PBC) using
XGBoost, with SHAP-based explainability for individual clinical decisions.

## Project Motivation

This is project #6 in a health-domain ML portfolio, following diabetes
classification, telco churn, chronic kidney disease, hospital readmission, and
imbalanced stroke prediction. This project introduces two new techniques not
covered previously:

1. **Gradient Boosting** (XGBoost) — a sequential ensemble method, as opposed
   to the bagging-based Random Forest used in earlier projects.
2. **SHAP (SHapley Additive exPlanations)** — game-theory-based explainability
   that quantifies each feature's contribution to *individual* predictions,
   not just global importance.

## Dataset

**Source:** UCI Machine Learning Repository — Cirrhosis Patient Survival
Prediction (Mayo Clinic PBC trial, 1974–1984), ~418 patients, 18 baseline
clinical/demographic features.

**Target:** Binary — did the patient die (`Status == 'D'`) during the
observation period, vs. survive or receive a liver transplant
(`Status in ['C', 'CL']`)? Class balance: ~38.6% death / ~61.4% non-death
(mild imbalance).

## Key Data Quality Finding: Structural Missingness

Six columns (`Drug`, `Ascites`, `Hepatomegaly`, `Spiders`, `Alk_Phos`, `SGOT`)
share **exactly 106 missing values**, and this missingness is **100%
explained** by whether a patient participated in the study's randomized drug
trial arm — patients recruited outside the trial simply never had these
measurements taken. Three other columns (`Cholesterol`, `Copper`,
`Tryglicerides`) show a **mixed** pattern: 106 missing values from the same
structural cause, plus a small number of additional missing values from
unrelated lab-processing issues.

This is a textbook case of **Missing Not At Random (MNAR)** with a known,
verifiable structural cause — verified empirically (not just assumed) by
checking that missingness in these columns perfectly co-occurs with missing
`Drug` values.

**Why this matters for modeling:** standard median/mode imputation implicitly
assumes missingness is random. Here it isn't — so two parallel preprocessing
pipelines were built:

- **Traditional pipeline** (for Logistic Regression / Random Forest baseline):
  median imputation, fit only on the training set.
- **Native pipeline** (for XGBoost): missing values are left as `NaN` and
  handled internally by the model's native missing-value routing, preserving
  the "this patient wasn't in the trial" signal implicitly through the
  `Drug_nan` one-hot column (created via `pd.get_dummies(..., dummy_na=True)`).

An explicit `in_trial` flag was also considered, but dropped after recognizing
it was **redundant** with `Drug_nan` — keeping both would have split feature
importance/SHAP attribution across two columns carrying identical
information, muddying the explainability story.

## Leakage Checks

- **`N_Days`** (days from enrollment to event/censoring) was **excluded** —
  it is only known *after* the outcome occurs (temporal leakage), unlike
  baseline labs which are available at prediction time.
- **`Stage`** (histological severity, 1–4, from biopsy) was **kept** — unlike
  `N_Days`, it is measured at baseline/diagnosis, and its correlation with
  mortality risk reflects genuine clinical signal, not leakage.

## Modeling & Model Selection

| Metric (5-fold CV, PR-AUC) | Logistic Regression | XGBoost (default) | XGBoost (tuned) |
|---|---|---|---|
| Mean | 0.763 | 0.685 | 0.752 |
| Std | 0.099 | — | — |

| Metric (held-out test set) | Logistic Regression | XGBoost (tuned) |
|---|---|---|
| Recall (death class) | 0.75 | 0.81 |
| Precision (death class) | 0.86 | 0.87 |
| ROC-AUC | 0.896 | 0.913 |
| PR-AUC | 0.851 | 0.902 |

**Honest finding:** with default hyperparameters, XGBoost *underperformed* the
simple Logistic Regression baseline — a direct consequence of the dataset's
small size (334 training rows) relative to the model's capacity, causing
overfitting. After hyperparameter tuning via `GridSearchCV` (shallow trees,
low learning rate, L1/L2 regularization, row subsampling), the cross-validated
performance gap nearly closed (0.752 vs. 0.763 — within one standard
deviation of each other). The held-out test set showed XGBoost ahead, but
given the small test size (84 rows) and the CV results, **the two models are
considered statistically comparable** in predictive performance.

**Final model chosen: XGBoost (tuned).** Not because it "won" decisively, but
because performance is competitive with LR while offering (a) native
handling of structurally-missing data without imputation bias, and (b)
individual-level explainability via SHAP — the primary goal of this project.

Best hyperparameters: `learning_rate=0.01, max_depth=3, n_estimators=100,
reg_alpha=5, reg_lambda=5, subsample=0.7`.

## SHAP Explainability Findings

**Global importance (summary plot):** `Bilirubin`, `Prothrombin`, and `Age`
are the top three predictive features — all clinically sensible markers of
liver dysfunction. Structural-missingness artifacts (`Drug_nan`, `Ascites`,
`Hepatomegaly`, `Spiders`) show minimal SHAP impact, which is reassuring: the
model is not relying on missingness patterns as a shortcut.

**Dependence plots (threshold effects):**
- **Bilirubin:** sharp non-linear threshold around ~5 mg/dL — below this,
  small increases have large impact; above it, the effect **plateaus**, even
  as values climb to 25 mg/dL. A linear model cannot capture this saturation.
- **Age:** threshold around 43–45 years, with a plateau after ~55 years —
  consistent with PBC typically becoming clinically severe in middle age
  rather than risk increasing linearly with age. (Note: `Age` is stored in
  **days** in the raw dataset — always convert to years for interpretation.)
- **Albumin:** an inverse, near-linear relationship — low albumin (impaired
  liver protein synthesis) pushes toward higher death risk.

**Force plots (individual explanations):** two contrasting patients with
similarly elevated `Alk_Phos` received opposite predictions, because one had
normal `Bilirubin`/`Prothrombin` (predicted survive) while the other had
additional elevated `SGOT`, high `Bilirubin`, and older age (predicted
death). This demonstrates SHAP's core value: no single abnormal lab value
determines risk — it's the **combination** that matters, and SHAP can make
that combination transparent to a non-technical stakeholder (e.g. a
clinician).

## Limitations

- Small dataset (418 patients total) limits statistical confidence in any
  single train/test split; cross-validation was used throughout to mitigate
  this, but variance across folds remains non-trivial (std ≈ 0.10 in PR-AUC).
- Structural missingness tied to trial participation means the model may not
  generalize well to populations recruited under different clinical trial
  designs.
- `Stage` prediction (multi-class, 4 classes with as few as 21 samples in the
  minority class) was considered as a bonus extension but requires
  additional care (e.g. class grouping) given the very small per-class
  sample sizes — flagged here as a known open direction rather than
  attempted without appropriate caveats.

## Files

- `predict.py` — `CirrhosisRiskPredictor` class; replicates training
  preprocessing exactly (binary/ordinal encoding, one-hot Drug with
  `dummy_na=True`, native NaN passthrough) and loads the saved XGBoost model
  for inference.
- `requirements.txt` — dependencies.
- `xgb_cirrhosis_model.pkl` — trained model (save via `joblib.dump(best_xgb,
  "xgb_cirrhosis_model.pkl")` after training; not included in this repo by
  default).

## Usage

```python
from predict import CirrhosisRiskPredictor

predictor = CirrhosisRiskPredictor(model_path="xgb_cirrhosis_model.pkl")

result = predictor.predict({
    "Age": 21464, "Sex": "F", "Ascites": "N", "Hepatomegaly": "Y",
    "Spiders": "N", "Edema": "N", "Bilirubin": 1.4, "Cholesterol": 261,
    "Albumin": 3.7, "Copper": 76, "Alk_Phos": 1718, "SGOT": 137.95,
    "Tryglicerides": 172, "Platelets": 190, "Prothrombin": 10.6,
    "Stage": 3.0, "Drug": "D-penicillamine"
})
print(result)
# {'risk_probability': 0.xx, 'predicted_label': 0 or 1, 'label_name': '...'}
```
