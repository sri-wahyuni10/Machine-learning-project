# Stroke Risk Prediction — Advanced Imbalanced Learning

Predicting stroke risk from patient health records, with a focus on handling extreme class imbalance (~19.5:1). This project builds on prior work in the same health-domain portfolio series, moving from moderate imbalance (hospital readmission) to a much more severe imbalance scenario, and comparing multiple imbalance-handling strategies with domain-informed model selection.

## Problem Statement

Stroke is a medical emergency where early risk identification matters more than overall predictive accuracy. In this dataset, only 4.9% of patients had a stroke — a naive model that always predicts "no stroke" would achieve 95% accuracy while being clinically useless. The goal of this project is to build a model that reliably flags at-risk patients (high recall on the stroke class), while critically evaluating the trade-offs involved.

## Dataset

- **Source:** Stroke Prediction Dataset (Kaggle), 5,110 patient records
- **Target:** `stroke` (binary) — class imbalance ratio ~19.5:1
- **Features:** demographic (age, gender, marital status, residence, work type), clinical (hypertension, heart disease, avg. glucose level, BMI), and behavioral (smoking status)

## Key EDA Findings

- **Age** shows the strongest separation between classes (mean age 67.7 for stroke patients vs. 42.0 for non-stroke).
- **Avg. glucose level** is also notably higher in stroke patients (132.5 vs. 104.8).
- **Missing `bmi` values (3.9% of records) are not random.** Patients with missing BMI had a stroke rate of 19.9% — roughly 4x the overall base rate of 4.9%. This is treated as a clinically meaningful signal rather than noise, and encoded as an explicit `bmi_missing` flag feature rather than being silently imputed away.
- **`smoking_status = 'Unknown'`** also carries signal, but in the opposite direction — a lower-than-average stroke rate (3.0% vs. 4.9% baseline), likely because this category is dominated by younger patients not asked about smoking history.

## Preprocessing Pipeline

- `bmi_missing` flag created **before** the train/test split (row-wise, rule-based — no leakage risk).
- Median imputation for `bmi`, one-hot encoding for nominal categoricals, ordinal (binary) encoding for two-level categoricals, and standard scaling for continuous numeric features — all fit **only on the training set**, wrapped in a single `ColumnTransformer` to guarantee train/test consistency.
- Stratified train/test split (80/20) — critical here given the small absolute number of positive cases (~250 total), where a non-stratified split risks producing an unrepresentative test set.

## Imbalance-Handling Techniques Compared

A baseline model with no imbalance handling was trained first to demonstrate the problem concretely:

| Approach | Recall (stroke) | Precision (stroke) | Accuracy |
|---|---|---|---|
| Baseline (no handling) | 0.02 | 1.00 | 0.95 |
| `class_weight='balanced'` | 0.80 | 0.14 | 0.76 |
| SMOTE (oversampling) | 0.80 | 0.14 | 0.75 |
| Balanced Random Forest | 0.74 | 0.16 | 0.79 |

The baseline confirms the core problem: high accuracy but a model that misses 98% of actual stroke cases — clinically unacceptable for a screening tool.

Note: SMOTE was implemented using `imblearn.pipeline.Pipeline` (not scikit-learn's `Pipeline`), which ensures resampling is correctly scoped to the training fold only, avoiding synthetic-data leakage into the test set.

## Model Selection

**Final model: Logistic Regression with `class_weight='balanced'`.**

`class_weight` and SMOTE produced statistically identical results (recall 0.80, precision 0.14), while Balanced Random Forest traded recall for a modest accuracy/precision gain — a trade-off that runs counter to this project's priority on recall. Given equivalent performance, `class_weight` was preferred over SMOTE for its lower complexity (no synthetic data generation, no additional dependency, no added leakage surface) and better interpretability via linear coefficients — a parsimony-driven decision in the spirit of Occam's razor.

## Threshold Tuning

Precision-recall curve analysis was performed across the full threshold range. Lowering the decision threshold from 0.5 to 0.3 was tested explicitly:

| Threshold | Recall | Precision | Accuracy |
|---|---|---|---|
| 0.5 (default) | 0.80 | 0.14 | 0.76 |
| 0.3 | 0.84 | 0.10 | 0.63 |

The marginal recall gain (+4 points) did not justify the accuracy and precision cost. **Conclusion: the default 0.5 threshold was retained**, since `class_weight` alone had already shifted the model close to an acceptable operating point at the algorithm level — additional threshold adjustment was not warranted.

## Evaluation Beyond Accuracy

**PR-AUC = 0.283**, compared to a random-guess baseline of 0.049 (the positive class rate) — roughly **5.8x better than chance**. PR-AUC was used instead of ROC-AUC because ROC-AUC's False Positive Rate term is inflated by the large number of true negatives in extreme imbalance settings, making it an overly optimistic and less informative metric here.

## Feature Interpretation & a Documented Limitation

Logistic Regression coefficients (on scaled features) largely align with clinical expectation: `age` (+1.88), `hypertension` (+0.59), `heart_disease` (+0.23), and `avg_glucose_level` (+0.20) all increase predicted stroke risk, consistent with established risk factors.

One coefficient required critical scrutiny: `work_type_children` showed a surprisingly high positive coefficient (+0.64). Investigation revealed this category contains only 2 stroke cases out of 687 records (0.29%) — the coefficient is very likely a statistical artifact of an extremely small positive sample, not a generalizable clinical pattern, and should not be interpreted as "being a child increases stroke risk." This is flagged explicitly as a limitation of linear coefficient interpretation on sparsely-represented subgroups, particularly under imbalance-handling techniques that push the model to seek any separating signal for the minority class.

## Files

- `stroke_prediction.ipynb` — full analysis notebook (EDA, preprocessing, modeling, evaluation)
- `predict.py` — standalone inference script, takes raw patient data as a dictionary and returns stroke risk probability
- `stroke_model.pkl` — serialized trained pipeline (preprocessing + model)
- `requirements.txt` — dependencies
- `README.md` — this file

## Key Takeaways

- Extreme imbalance (19.5:1) requires intervention beyond post-hoc threshold tuning (which was sufficient in a prior, less-imbalanced project) — algorithm-level or data-level techniques become necessary.
- More complex techniques (SMOTE, ensemble resampling) do not automatically outperform simpler ones (`class_weight`) — empirical comparison, not assumption, should drive the choice.
- Missingness in medical data can be informative rather than noise, in both directions (higher and lower risk).
- Model coefficients require scrutiny for subgroup sample size before being trusted as clinically meaningful — a small positive-class count within a subgroup can produce a misleadingly large coefficient.
