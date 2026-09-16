penjelasan tentang kolom dataset
Kolom	Arti	               Kolom	Arti
age	---usia	                    pcv	---packed cell volume
bp	----tekanan darah	        wc	----jumlah sel darah putih
sg	----specific gravity urin	rc	----jumlah sel darah merah
al	----albumin (urin)	        htn	----hipertensi
su	----gula (urin)	            dm	----diabetes mellitus
rbc	----sel darah merah (urin)	cad	----penyakit jantung koroner
pc	----pus cell	            appet	-nafsu makan
pcc	----pus cell clumps	        pe	----pedal edema (bengkak kaki)
ba	----bakteri	                ane	----anemia
bgr	----gula darah acak	        sc	----serum creatinine
bu	----blood urea	            sod	----sodium
pot	----potassium	            hemo	-hemoglobin


"Hyperparameter tuning tidak dilakukan karena baseline Logistic Regression sudah mencapai recall sempurna (1.0 ± 0.0) di 5-fold cross-validation; ruang untuk perbaikan lebih lanjut lewat tuning tidak ada.

# Chronic Kidney Disease Prediction

A supervised machine learning project that predicts Chronic Kidney Disease (CKD) from clinical lab and examination data. This is the third project in a health-focused ML portfolio series (following a diabetes classification project and a telco churn project), extending the domain narrative into a dataset with heavier data-quality issues and clinically meaningful missingness patterns.

## Table of Contents
- [Dataset](#dataset)
- [Problem Framing](#problem-framing)
- [Data Cleaning](#data-cleaning)
- [Missingness Analysis](#missingness-analysis)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Modeling & Evaluation](#modeling--evaluation)
- [Critical Investigation of Results](#critical-investigation-of-results)
- [Results Summary](#results-summary)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [How to Run Inference](#how-to-run-inference)

## Dataset

- **Source**: [UCI Machine Learning Repository — Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Size**: 400 patients, 24 clinical features + 1 target (`classification`)
- **Target distribution**: 250 CKD (62.5%) vs. 150 not-CKD (37.5%) — moderately imbalanced
- **Features**: a mix of continuous lab measurements (e.g. blood urea, serum creatinine, hemoglobin), discrete/ordinal clinical readings (e.g. specific gravity, albumin, sugar), and binary clinical indicators (e.g. hypertension, diabetes mellitus, anemia)

## Problem Framing

In a medical screening context, a **false negative** (predicting a CKD patient as healthy) is far more costly than a false positive — a missed diagnosis delays treatment for a progressive disease, while a false positive only leads to unnecessary follow-up testing. This project treats **recall on the CKD class** as the primary evaluation metric rather than raw accuracy.

## Data Cleaning

Raw data quality issues found and addressed:

- **Hidden whitespace/tab characters**: several columns that should have been numeric (`pcv`, `wc`, `rc`) were loaded as `object` dtype due to stray tab characters (e.g. `'\t43'`, `'\t?'`). Values were stripped of whitespace before being converted with `pd.to_numeric(errors='coerce')`, which preserves genuinely valid numeric readings while turning true placeholders (`'?'`) into proper missing values.
- **Inconsistent categorical labels**: binary categorical columns (`dm`, `cad`) contained duplicate categories caused by whitespace (e.g. `'yes'` vs `'\tyes'` vs `' yes'`). A blanket strip was applied across all object-dtype columns to avoid one-hot encoding from treating identical categories as distinct.
- **Corrupted target labels**: the target column contained `'ckd'`, `'ckd\t'`, and `'notckd'` — effectively 3 labels for 2 real classes. Left uncleaned, this would have broken stratified splitting and misrepresented the confusion matrix. Cleaned to exactly 2 classes.

## Missingness Analysis

Missing values are not uniformly distributed — they are heavily concentrated within the CKD class:

| Column | Missing (CKD, n=250) | Missing (not-CKD, n=150) |
|---|---|---|
| rbc | 143 | 9 |
| pcv | 67 | 4 |
| wc | 99 | 7 |
| rc | 124 | 7 |
| sod/pot | ~80+ | ~7 |

This is clinically explainable: lab tests like RBC count and packed cell volume are typically ordered when a physician already suspects renal impairment, while patients who appear healthy skip these tests. The pattern was noted explicitly because "was this test even ordered" is itself an informative — and potentially leaky — signal that deserves scrutiny rather than being silently absorbed by generic imputation.

## Preprocessing Pipeline

Features were grouped into three types, each handled by its own `SimpleImputer` + transformer inside a `ColumnTransformer`, verified against `dtype` and `nunique()` rather than assumed from column names alone:

| Group | Columns | Imputation | Further transform |
|---|---|---|---|
| Continuous numeric | age, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc | median (robust to lab outliers) | `StandardScaler` |
| Ordinal / discrete | sg, al, su, bp | median | `StandardScaler` |
| Binary categorical | rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane | most frequent | `OneHotEncoder(drop='if_binary')` |

Median (not mean) was used for numeric groups because lab values such as white cell count and creatinine are prone to extreme outliers in severely ill patients, which would pull the mean away from a representative value.

Train/test split (80/20, stratified) was performed **before** any imputation to prevent statistics from the test set leaking into imputation values.

## Modeling & Evaluation

Four classifiers were compared using 5-fold stratified cross-validation on the training set only (recall scoring), keeping the test set fully untouched during model selection to avoid selection leakage:

| Model | Mean Recall (CV) | Std |
|---|---|---|
| **Logistic Regression** | **1.0000** | 0.0000 |
| Random Forest | 0.9950 | 0.0100 |
| SVM | 0.9850 | 0.0122 |
| KNN | 0.9500 | 0.0158 |

Logistic Regression was selected as the final model.

## Critical Investigation of Results

A perfect cross-validated recall score is a red flag, not an automatic win — it was investigated rather than taken at face value:

1. **Train/test leakage check**: confirmed no overlapping indices and no duplicate rows between train and test sets.
2. **Coefficient inspection**: the top contributing features (`sg`, `hemo`, `al`, `pcv`, `htn`, `dm`, `sc`, `appet`) are all clinically recognized CKD indicators, with no single feature dominating disproportionately — evidence against a spurious shortcut feature.
3. **Cross-validation stability**: recall stayed at or near 1.0 across all 5 folds (`[1.0, 1.0, 0.96, 1.0, 1.0]`), not just one lucky split.

Conclusion: this dataset is structurally easy to separate (likely because the CKD cases represent clearly progressed disease), which explains the near-perfect scores without indicating a pipeline bug.

## Results Summary

Final Logistic Regression performance on the held-out test set (n=80):

```
                precision   recall  f1-score
  notckd (0)       1.00      1.00      1.00
     ckd (1)       1.00      1.00      1.00
```

Confusion matrix: `[[30, 0], [0, 50]]` — zero misclassifications.

Hyperparameter tuning was deliberately **not performed**: with cross-validated recall already at a perfect 1.0 ± 0.0, there is no headroom left for improvement, and running a grid search would only confirm that the default parameters are already optimal.

## Limitations

- This dataset is small (400 rows) and appears structurally "clean" in how CKD vs. non-CKD cases were sampled. A 100% recall here should **not** be read as "this model detects CKD with 100% accuracy in clinical practice" — real-world early-stage or ambiguous cases are almost certainly harder to classify than what this dataset represents.
- Missingness itself correlates strongly with the target (see [Missingness Analysis](#missingness-analysis)), which may make performance on this specific dataset optimistic compared to a deployment setting where test-ordering patterns differ.
- The model has not been validated against an external/independent CKD dataset.

## Repository Structure

```
├── ckd_analysis.ipynb        # full exploration, cleaning, modeling notebook
├── ckd_model.pkl             # trained Pipeline (preprocessing + Logistic Regression)
├── ckd_label_mapping.json    # {"ckd": 1, "notckd": 0} label mapping
├── predict.py                # standalone inference script
├── requirements.txt
└── README.md
```

## How to Run Inference

```bash
pip install -r requirements.txt
python predict.py
```

`predict.py` loads `ckd_model.pkl` and `ckd_label_mapping.json`, accepts new patient data as a single-row DataFrame (24 features matching the training columns), and outputs both the predicted class (`ckd` / `notckd`) and the associated probability.
