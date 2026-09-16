# Synthetic Indonesian Retail Sales Dataset & Forecasting

A Rossmann-style retail sales dataset **generated entirely from scratch** — designed, coded, and validated end-to-end, then used to train and debug a LightGBM sales forecasting model. Unlike working from a pre-cleaned Kaggle dataset, this project required designing the data generating process (DGP) itself: deciding what patterns should exist in the data and why, then verifying afterward that both the data and the model actually reflect those decisions.

## Why build a synthetic dataset instead of using Rossmann directly?

Working with real-world datasets teaches cleaning and feature engineering, but it hides the ground truth — you never really know *why* a value is what it is. Building the data generating process from scratch flips that: every pattern in the data has a known, intentional cause, which makes it possible to rigorously verify whether feature engineering and modeling choices are actually capturing the right signal — including catching real pipeline bugs along the way (see [Key Findings](#key-findings-a-real-pipeline-bug) below).

## Dataset Design

- **Scale:** 30 stores × 912 days (Jan 2022 – Jun 2024) = 27,360 rows
- **Structure:** two tables, mirroring the original Rossmann schema
  - `store.csv` — static store attributes (StoreType, Assortment, CompetitionDistance, Promo2, etc.)
  - `train.csv` — daily time series (Sales, Customers, Open, Promo, StateHoliday, etc.)
- **Localization:** rather than reusing Christmas-driven seasonality, sales are driven by Ramadan/Lebaran shopping patterns — a surge in the week before Lebaran, followed by a hard drop (stores closed) on the holiday itself.

### Injected patterns (the "ground truth")

| Component | Description |
|---|---|
| Trend | Gradual ~25% baseline growth from start to end of the period, per store |
| Weekly seasonality | Weekday/weekend multiplier (Saturday busiest, weekdays quieter) |
| Promo effect | Multiplicative boost, magnitude varies by `StoreType` (interaction effect) |
| Lebaran effect | 1.8x sales multiplier in the 7 days before Lebaran; stores closed (Sales=0) on the holiday itself |
| Structured missingness | `CompetitionDistance` is MNAR at the store level (missing for stores without a recorded competitor), not missing at random |
| Target leakage (intentional) | `Customers` is generated as a noisy function of `Sales`, deliberately included as a leakage trap for feature-selection practice |
| Random anomalies | ~0.3% of rows have Sales cut to 10-30% of their expected value, with no signal in any other column — simulating stockouts or system outages |

## Pipeline

1. **Generation** — `store.csv`, `calendar.csv` (date scaffold + Lebaran/holiday calendar), `train.csv` (cross join of stores × dates, then sales computed from the formula above)
2. **Merge & missingness audit** — confirmed `CompetitionDistance` missingness is structural (whole-store, not row-level)
3. **Feature engineering** — derived boolean flags (`IsPromo2ActiveOnThisDate`, `IsCompetitorActive`, `IsMonthInPromoInterval`) instead of dropping MNAR columns; `EffectiveCompetitionDistance` combines distance with activation status
4. **EDA** — verified injected trend, Lebaran spikes, and StoreType ordering are visible in aggregate plots
5. **Lag/rolling features** — `Sales_lag_7`, `Sales_rolling_mean_7`, computed **before** dropping closed-store rows to preserve correct time gaps, and shifted to avoid leaking same-day sales
6. **Time-based train/test split** — train on 2022-2023, test on Jan-Jun 2024 (deliberately includes the 2024 Lebaran window as a stress test)
7. **Modeling** — LightGBM regressor with native categorical support
8. **Error analysis & SHAP** — used residual analysis to find a real pipeline bug, then SHAP to validate the model learned the correct causal structure

## Key Findings: A Real Pipeline Bug

The first trained model had a large residual error concentrated almost entirely around the Lebaran windows. Investigating the worst predictions revealed the cause: `IsLebaranShoppingWindow` was computed and used to *generate* the sales formula, but was never carried over into the final `train.csv` — the model had no way to know why sales spiked. This was a silent bug: the code ran without errors and produced plausible-looking output at every step.

Merging the missing flag back in from `calendar.csv` and retraining:

| Metric | Before fix | After fix |
|---|---|---|
| MAE | 670.74 | 466.33 |
| RMSE | 1195.21 | 631.49 |

After the fix, remaining large residuals were traced to the intentionally injected random anomalies (0.3% of rows) — an irreducible error by design, since those rows carry no predictive signal in any feature. Confirming this distinction (fixable pipeline bug vs. irreducible noise) was done through residual inspection, not assumption.

## SHAP Validation

Feature importance broadly matched the causal structure of the generating formula: `DayOfWeek`, `StoreType`, `Promo`, `Sales_lag_7`, and `IsLebaranShoppingWindow` ranked highest — consistent with the weekly seasonality, base sales level, promo boost, and Lebaran effect built into the formula. Features that were present in `store.csv` but never actually wired into the sales formula (`IsCompetitorActive`, `IsPromo2ActiveOnThisDate`) correctly showed near-zero importance — a useful confirmation that SHAP surfaces engineered features with no real causal link to the target, rather than just their apparent complexity.

## Stack

Python, pandas, numpy, LightGBM, SHAP, matplotlib

## Files

- `store.csv`, `calendar.csv`, `train.csv` — generated dataset
- `predict.py` — inference wrapper that replicates the full training-time feature engineering pipeline
- `save_model_artifacts.py` — helper to export the trained model + metadata from the training notebook

## Possible Extensions

- Store-level model residual analysis (which stores are hardest to predict, and why)
- Hyperparameter tuning
- Additional interaction features (e.g. `StoreType` x `DayOfWeek`)
- FastAPI deployment wrapper around `predict.py`
