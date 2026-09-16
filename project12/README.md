# German Credit Risk Classification

Binary classification project predicting credit risk (`good` vs `bad`) using the German Credit dataset (Statlog, UCI). This project emphasizes cost-sensitive decision making using an asymmetric cost matrix, in addition to standard EDA, MNAR investigation, and encoding strategy design.

## Dataset

- **Source**: [UCI Machine Learning Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **File used**: `german.data` (raw symbolic-coded categorical version, not `german.data-numeric`)
- **Size**: 1000 rows, 20 features + 1 target
- **Target**: `class` — `good` (700 rows, 70%) vs `bad` (300 rows, 30%) credit risk

## Problem Framing

Unlike most prior classification projects in this portfolio, this dataset comes with an **official asymmetric cost matrix** from UCI: misclassifying a `bad` applicant as `good` (approving a risky loan) costs **5x** more than misclassifying a `good` applicant as `bad` (rejecting a safe applicant). This shifted the modeling objective away from standard accuracy/F1 optimization toward **cost-based threshold selection**.

## Key Techniques

### 1. MNAR Investigation
Three categorical columns (`checking_status`, `savings_status`, `property_magnitude`) contain explicit "unknown/no account" categories rather than `NaN` values. Groupby-against-target analysis revealed **inconsistent risk direction** across these columns:
- `checking_status` / `savings_status`: "no account" categories correlate with **lower** bad-credit risk
- `property_magnitude`: "unknown/no property" correlates with **higher** bad-credit risk

This demonstrates that MNAR patterns cannot be assumed to behave uniformly even within the same dataset — each column requires independent verification.

### 2. Ordinal vs Nominal Encoding Strategy
- **Ordinal (data-driven order)**: `checking_status`, `savings_status`, `property_magnitude` — category order derived from empirical bad-rate ranking, not administrative/nominal value order
- **Ordinal (domain-logic order)**: `employment`, `credit_history`, `job` — category order derived from clear natural hierarchy (tenure length, credit history severity, skill level)
- **Nominal (one-hot)**: `purpose`, `housing`, `personal_status`, `other_parties`, `other_payment_plans`, `own_telephone`, `foreign_worker` — no natural ordering

### 3. Cost-Sensitive Threshold Tuning
Standard 0.5 threshold was compared against a cost-minimizing threshold search:

```
total_cost = (False Negatives × 5) + False Positives
```

Optimal threshold found via sweep: **0.15** (cost = 98, vs cost = 186 at default 0.5 threshold).

| Threshold | Recall (bad) | Precision (bad) | Accuracy | Total Cost |
|---|---|---|---|---|
| 0.50 (default) | 0.47 | 0.52 | 0.71 | 186 |
| 0.40 | 0.57 | 0.52 | 0.71 | 162 |
| 0.35 | 0.68 | 0.52 | 0.71 | 133 |
| 0.30 | 0.72 | 0.49 | 0.69 | 129 |
| **0.15 (optimal)** | **0.90** | **0.44** | **0.63** | **98** |

At the cost-optimal threshold, accuracy and precision drop noticeably compared to the default threshold. This is an intentional trade-off, not a modeling failure — standard metrics assume symmetric error costs, which doesn't hold here.

## Pipeline

```
Raw data (symbolic codes) 
  → Human-readable mapping (dict-based .map())
  → train_test_split (stratified)
  → ColumnTransformer:
      - OrdinalEncoder (6 columns, explicit category order)
      - OneHotEncoder (7 columns, handle_unknown='ignore')
      - StandardScaler (7 numeric columns)
  → LogisticRegression (max_iter=1000)
  → Threshold tuning via predict_proba + cost function
```

## Inference

`CreditRiskPredictor` class handles:
- Raw symbolic-code input → readable mapping conversion
- Validation guard against unrecognized category codes (raises `ValueError` instead of silently producing `NaN`)
- Custom threshold (0.15) applied to `predict_proba` output
- Batch prediction support (vectorized, no per-row loop)

Validated on the full held-out test set via two independent paths (direct pipeline call vs. `CreditRiskPredictor` class) producing identical classification reports — confirming no logic bugs in the inference wrapper.

## Results Summary (threshold = 0.15)

```
              precision    recall  f1-score   support
         bad       0.44      0.90      0.59        60
        good       0.92      0.51      0.66       140
    accuracy                           0.63       200
```

## Known Limitations / Next Steps

- Only Logistic Regression tested; tree-based models (Random Forest, XGBoost) not yet compared for potential lower cost at similar or better recall
- SHAP interpretability not yet applied to this dataset
- `personal_status` column conflates marital status and gender into a single categorical field — a known fairness/representation concern in this dataset that was not addressed in modeling (left as-is per original dataset structure, flagged for awareness)

## Tech Stack

Python, pandas, scikit-learn (`ColumnTransformer`, `OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`, `LogisticRegression`), joblib
