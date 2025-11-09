# Booking Status Prediction (Master 2 project)

End-to-end coursework project for the Kaggle Booking Cancel Prediction competition. The goal is to forecast the final status of each hotel booking (check-out, cancel, or no-show) while squeezing the last decimals of the weighted F1 metric.

## Competition Overview
- Kaggle competition: [Booking Cancel Prediction](https://www.kaggle.com/competitions/booking-cancel-prediction)
- Context: hotel operators want to anticipate cancellations and no-shows to protect revenue and staffing decisions.
- Objective: predict the `reservation_status` for every row in the hidden test split. Labels are encoded as `0 = Check-out`, `1 = Cancel`, `2 = No-Show`.
- Evaluation: weighted F1 score computed one-vs-all and averaged using the class frequencies (`sklearn.metrics.f1_score(..., average="weighted")`).
- Features: the dataset mirrors the well-known Hotel Booking Demand data (columns such as `hotel`, `lead_time`, arrival date triplets, guests counts, room types, deposit type, ADR, special requests, etc.). The Kaggle data tab contains the authoritative description provided in the course notes.

## Repository Layout
- `src/model/train.py`: main training entry point (feature engineering, CV, optional hyperparameter search, final fit, and test inference).
- `src/model/data_processing.py`: custom `DataFrameTransformer` that handles feature engineering (hotel/country flags, arrival sine/cosine encoding, parking and room mismatch indicators, ADR clipping, etc.).
- `src/data/`: expected location for `train_data.csv`, `test_data.csv`, and derived full or transformed CSVs.
- `notebooks/`: exploratory analysis and sandbox experiments.

## Setup
1. Install Python 3.12+ (the project ships with a `pyproject.toml` and `uv.lock`).
2. Create the environment and install dependencies with:
   ```bash
   uv sync
   ```

## Workflows
- Standard training (cross-validation + final fit + inference):
  ```bash
  uv run python -m src.model.train \
    --model xgboost \
    --n-splits 5 \
    --random-state 3
  ```
  This writes:
  - `models/booking_status_pipeline.joblib`
  - `artifacts/training_metrics.json` (fold scores, OOF confusion matrix, classification report)
  - `artifacts/test_predictions.csv` (ready for submission)
- Add a randomized search before CV (tunes the pipeline with stratified folds and the same weighted-F1 metric):
  ```bash
  uv run python -m src.model.train \
    --model catboost \
    --search \
    --search-iterations 25 \
    --search-n-splits 4
  ```
  Use `--search-n-jobs -1` to parallelize, or `--no-save` to skip persisting artefacts.

## Modeling Approach
- Feature engineering: `DataFrameTransformer` builds leakage-safe flags tailored to the dataset—city-vs-resort indicator, home-country flag (`PRT`), replacement of undefined distribution channels, room mismatch indicator, parking and week-number flags, cyclical encoding of arrival dates, integer coercion of count columns, and optional dropping of high-ADR outliers (default cap: 1000) so that the downstream models stay robust.
- Preprocessing: numerical columns go through median imputation + standard scaling, categoricals are imputed with the most frequent value and one-hot encoded. CatBoost skips the ColumnTransformer and consumes the engineered DataFrame directly.
- Models: choose between `histgb`, `xgboost`, and `catboost`. Each option ships with sensible defaults plus a dedicated randomized-search space. All fits use class-balanced sample weights to compensate for the natural imbalance between check-outs, cancellations, and no-shows.
- Evaluation: stratified K-fold CV reports per-fold weighted F1, its mean and standard deviation, confusion matrix, and a full classification report. The script then refits on the full training data and generates leaderboard-ready predictions while ensuring no rows disappear during transformation (test ADRs above the cap are clipped, not removed).
- Artefacts: every run logs the configuration, tuned hyperparameters (if any), metrics, and predictions under `artifacts/`, making it trivial to track experiments or reproduce results later.
