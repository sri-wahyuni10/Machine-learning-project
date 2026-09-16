# Bank Marketing Term Deposit Classification

Binary classification project predicting whether a bank customer will subscribe to a term deposit, built to practice preprocessing pipelines for mixed numerical + categorical data using scikit-learn's `Pipeline` and `ColumnTransformer`.

## Problem

Given customer demographic data, contact history, and macroeconomic indicators, predict whether a customer will subscribe to a term deposit (`y = yes/no`) as a result of a telemarketing campaign. The goal is to help a bank prioritize which customers to call.

## Dataset

- **Source:** [UCI Machine Learning Repository — Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **File used:** `bank-additional-full.csv`
- **Size:** 41,188 rows, 20 features + 1 target
- **Target distribution:** 11.27% `yes`, 88.73% `no` (imbalanced)

## Key Data Preparation Decisions

This project deliberately focused on preprocessing reasoning, not just running `.fit_transform()`. Key decisions:

| Issue | Finding | Treatment |
|---|---|---|
| Missing values | No explicit NaN; hidden as string `"unknown"` in categorical columns | Investigated per column before deciding treatment |
| `default` column | 20.87% `"unknown"`; subscribe rate for `unknown` (5.15%) vs `no` (12.88%) differs sharply | Treated as MNAR — kept as its own category, not imputed |
| `housing` / `loan` | Both 100% overlapping `"unknown"` rows (990 rows) | Kept as separate nominal categories; added a combined `loan_info_missing` flag for interpretability |
| `duration` | Mean duration for `y=yes` (553s) is 2.5x that of `y=no` (221s) — only known *after* the call ends | **Dropped** — target leakage, would make the model unusable for real prediction |
| `pdays` | 96.3% of rows use sentinel value `999` ("never contacted before") | Converted to binary flag `was_contacted_before`; raw column dropped |
| `education` | Has a natural order (`basic.4y` < ... < `university.degree`) | Encoded with `OrdinalEncoder` and an explicit category order, not `OneHotEncoder` |

## Pipeline Architecture

```
ColumnTransformer
├── numeric_pipeline      → StandardScaler
│     (age, campaign, previous, economic indicators, engineered flags)
├── nominal_pipeline      → OneHotEncoder(handle_unknown='ignore')
│     (job, marital, default, housing, loan, contact, month, day_of_week, poutcome)
└── ordinal_pipeline      → OrdinalEncoder(explicit category order)
      (education)
```

Train/test split (80/20, stratified) was performed **before** fitting any preprocessing step, to prevent data leakage from test statistics into training.

## Modeling & Evaluation

Because the target is imbalanced, **accuracy was not used as the primary metric**. PR-AUC and per-class precision/recall (focused on the minority `yes` class) were used instead.

| Model | PR-AUC | Precision (yes) | Recall (yes) | F1 (yes) |
|---|---|---|---|---|
| Logistic Regression (threshold 0.5) | 0.466 | 0.69 | 0.22 | 0.33 |
| Logistic Regression (tuned threshold = 0.203) | 0.466 | 0.458 | 0.593 | 0.516 |
| Random Forest (default params) | 0.392 | 0.46 | 0.49 | 0.47 |
| Random Forest (`max_depth=10`, `min_samples_leaf=20`) | **0.491** | – | – | – |
| **Random Forest tuned + threshold = 0.696** | **0.491** | **0.485** | 0.581 | **0.528** |

**Final model: Random Forest** (`max_depth=10`, `min_samples_leaf=20`, `class_weight='balanced'`, `n_estimators=200`) with a decision threshold of **0.696**, chosen to maximize F1-score on the minority class.

### Notable finding
A default Random Forest (`n_estimators=200`, no depth limit) actually **underperformed** Logistic Regression on PR-AUC, due to overfitting on the high-dimensional one-hot encoded feature space. Constraining tree depth and leaf size reversed this — a reminder that model comparisons with default hyperparameters can be misleading.

## What Would Come Next

- Systematic hyperparameter search (`RandomizedSearchCV`) instead of manual tuning
- SHAP analysis to interpret which features drive predictions
- LightGBM as an additional comparison model
- Business-cost-based threshold selection (cost of a false positive call vs. a missed lead) rather than pure F1 optimization

## Tech Stack

Python, pandas, scikit-learn (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `OrdinalEncoder`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`), NumPy

## Files

- `bank_marketing_classification.ipynb` — full analysis notebook
- `README.md` — this file
- `CATATAN_PROSES.md` — personal development notes (Indonesian)
