# Rossmann Store Sales Forecasting

Feature-based time series forecasting for daily sales across 1,115 Rossmann drug stores, using a supervised regression approach (gradient boosting) rather than classical statistical models (ARIMA/Prophet).

## Problem

Predict daily `Sales` per store, using historical sales, store metadata (competition, promotions), and calendar information — while strictly respecting temporal causality (no future information may leak into training).

## Approach

Rather than treating this as a classical univariate time series problem, sales forecasting is framed as **supervised regression**: each (store, date) pair becomes a row with engineered features, and a gradient boosting model (LightGBM) learns the mapping to `Sales`. This mirrors the approach used by most top solutions in the original Kaggle competition and is the dominant paradigm in industry forecasting today (see M5/Walmart competition winners).

### Feature Engineering
- **Lag features**: `Sales_lag_1/7/14` — computed per-store, sorted chronologically first to avoid cross-store contamination
- **Rolling statistics**: `Sales_roll_mean_7/std_7` — computed with `.shift(1)` before `.rolling()` to prevent leakage from the current day
- **Calendar features**: `DayOfWeek`, `Month`, `Year`, `WeekOfYear`, `IsWeekend`
- **Competition dynamics**: `competition_open_months` (duration since competitor opened), with an explicit `competition_open_since_unknown` flag for stores where this is genuinely unknown (not imputed with mean/median, which would fabricate a false signal)
- **Promo2 recurring campaigns**: parsed `PromoInterval` (e.g. `"Jan,Apr,Jul,Oct"`) into `is_promo2_active_month`, handling a non-standard `"Sept"` abbreviation found in the raw data

### Leakage Prevention
- `Customers` dropped entirely — not known at prediction time in a real deployment scenario
- **Time-based train/test split** (cutoff `2015-06-19`) instead of random split — random split would let the model "see" future dates during training via correlated lag features
- Rows with `Open == 0` excluded from training (deterministic zero-sales), but **retained** during lag/rolling computation to preserve calendar-day continuity

### Modeling
- **LightGBM regression**, target log-transformed (`log1p`) to stabilize variance and align with the percentage-based evaluation metric
- Native categorical support for `Store`, `StateHoliday`, `StoreType`, `Assortment` (no one-hot encoding needed)
- Hyperparameter tuning via `RandomizedSearchCV` with `TimeSeriesSplit` (expanding-window CV, never validating on data older than what it trained on)
- **Custom RMSPE scorer** for tuning — optimizing directly for the competition's evaluation metric rather than a generic RMSE, which measurably improved results over RMSE-optimized tuning

## Results

| Model | RMSPE |
|---|---|
| Baseline (default LightGBM) | 0.1148 |
| Tuned (RMSE scoring) | 0.1213 |
| **Tuned (custom RMSPE scoring)** | **0.1138** |

Notably, hyperparameter tuning provided only marginal improvement over the baseline — feature engineering (lag/rolling, promo parsing, competition duration) was the dominant driver of model performance, not tuning depth.

## Model Interpretation (SHAP)

- `Sales_lag_14` and `Sales_roll_mean_7` dominate predictions with a non-linear, saturating relationship (diminishing returns at high sales volumes)
- `Promo` shows the cleanest, most consistent effect of any feature, with a secondary interaction: promo impact is proportionally stronger on weekdays than weekends
- `Sales_lag_1` shows a non-monotonic (V-shaped) relationship — low values don't uniformly predict low future sales, since they often follow store closures/holidays and precede a rebound
- `Store` (despite dominating LightGBM's default feature importance) contributes relatively little to individual predictions in SHAP terms — a good illustration of how split-count-based importance can be misleading for high-cardinality categorical features

## Inference

`rossmann_predictor.py` provides an OOP wrapper (`RossmannSalesPredictor`) for production-style inference:
- Accepts store ID, target date, and a 14-day calendar-complete sales history
- Recomputes lag/rolling features from the provided history
- Validates that the history window is calendar-continuous (including closed days) — a mismatch here was found to bias predictions by 5-10 percentage points during testing
- Preserves training-time categorical definitions to avoid category mismatch errors

## Tech Stack
Python, pandas, LightGBM, scikit-learn, SHAP, joblib

## Project Files
- `notebook.ipynb` — full pipeline (EDA → feature engineering → modeling → SHAP)
- `rossmann_predictor.py` — inference wrapper
- `save_bundle.py` — script to bundle trained model + metadata for deployment
- `CATATAN_PROSES.md` — detailed development journal (Indonesian)

## Key Learnings
- Random train/test splits are unsafe for time series with lag features, even if leakage isn't immediately obvious
- Mismatches between CV scoring metric and final evaluation metric can silently produce a "worse" tuned model — always validate on the true target metric
- Native categorical support in gradient boosting libraries can crash on Arrow-backed pandas string dtypes; explicit `object` casting is a safer intermediate step
- NumPy/LightGBM ABI incompatibilities can produce low-level crashes (access violations) with no informative Python traceback — isolating environments per project prevents cross-project dependency conflicts
