# Tennis Match Duration Prediction

This project predicts **how long an ATP tennis match will last (in minutes)** using information that is available *before* the match starts (tournament, surface, round, best-of format, player rankings, age, height, etc.).

It compares two approaches:

1. **Baseline model:** predicts the average match duration depending only on whether the match is played best-of-3 or best-of-5.
2. **Machine learning model:** an H2O AutoML model trained on all preprocessed features.

The goal is to see how much a machine learning model improves on the simple baseline.

---

## Project structure

```
tennis_predictions/
├── README.md
├── .gitignore
└── tennis_pred_1/
    ├── Makefile                      # shortcuts for install, run and test
    ├── pyproject.toml                # project definition and dependencies
    ├── uv.lock                       # exact dependency versions (managed by uv)
    ├── .python-version
    ├── notebooks/
    │   └── Tennis_data_master_table.ipynb
    ├── src/
    │   └── tennis_pred_1/
    │       ├── __init__.py
    │       ├── ATP_tennis_group_project.csv               # raw dataset
    │       ├── Tennis_time_predictions_preprocessing.py   # data preparation
    │       ├── Tennis_time_predictions_baseline.py        # baseline model
    │       └── Tennis_time_predictions_modeling.py        # H2O AutoML model + comparison
    ├── tests/
    │   ├── test_Tennis_time_predictions_preprocessing.py
    │   ├── test_Tennis_time_predictions_baseline.py
    │   └── test_Tennis_time_predictions_modeling.py
```

---

## Data

The dataset comes from Kaggle: [ATP Tennis 2000 - 2026 Daily update](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) (Kaggle user *dissfya*), downloaded on **2026-10-01**.

It is stored in the repository as `src/tennis_pred_1/ATP_tennis_group_project.csv`. Since the Kaggle dataset is updated daily, a newer download may contain more matches and lead to slightly different results.

---

## Requirements

- **Python 3.11**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **make** (preinstalled on Linux, macOS and GitHub Codespaces)
- **Java** (required by H2O for the modeling script; it must be installed and available on the `PATH` or via `JAVA_HOME`)

---

## Getting started

All commands are run from the `tennis_pred_1` folder (where the `Makefile` and `pyproject.toml` are located):

```bash
cd tennis_pred_1
make install
```

This runs `uv sync` and installs all dependencies in the exact versions defined in `uv.lock`, including the project package itself and the development dependency `pytest`.

---

## Usage

| Command | What it does |
|---|---|
| `make install` | Installs all dependencies (`uv sync`) |
| `make baseline` | Runs the baseline model and prints its RMSE |
| `make model` | Trains the H2O AutoML model, prints its RMSE and the improvement over the baseline |
| `make test` | Runs all tests with pytest |

`make model` also runs the baseline (the modeling script imports the baseline result for the comparison), so it covers the complete pipeline.

---

## Pipeline

### 1. Preprocessing (`Tennis_time_predictions_preprocessing.py`)

This file only defines functions; it is used by the baseline and modeling scripts. The function `preprocessing_splitting(df)` runs all steps in this order:

1. **Drop columns:** removes IDs, metadata and all match statistics that are only known *after* the match (e.g. aces, break points, score) to avoid data leakage.
2. **Date → month:** converts `tourney_date` into the month and encodes it as sine and cosine, so that December and January are treated as close to each other.
3. **Fix names:** unifies known spelling variants of player names.
4. **Blind winner/loser:** randomly assigns winner and loser to `player_1` and `player_2`, so the model cannot learn that player 1 is always the winner.
5. **Remove duplicates**
6. **Handle missing values:**
   - removes matches with missing or 0 minutes,
   - missing rank → `9999` (unranked), missing rank points → `0`,
   - missing age and height are imputed with MICE (`IterativeImputer`).
7. **Encoding:**
   - label encoding for tournament ID/name, player names and nationality,
   - one-hot encoding for tournament level, surface and handedness,
   - ordinal encoding for the round (R128 → 1 … F → 7).
8. **Train/test split:** 80 % training, 20 % test (`random_state=42`), target variable `minutes`.

### 2. Baseline model (`Tennis_time_predictions_baseline.py`)

Calculates the mean match duration per `best_of` value (3 or 5 sets) and uses it as the prediction. The result is the RMSE on the test set.

### 3. Modeling (`Tennis_time_predictions_modeling.py`)

Trains an **H2O AutoML** regression model on the training data and evaluates the best model on the test set. Afterwards it compares the RMSE of the AutoML model with the baseline RMSE and prints the improvement in minutes and percent.

The training time is set via `max_runtime_secs` in `train_model` (currently 30 seconds for testing; longer runs such as 10 minutes usually give better results).

---

## Results (example run)

| Model | RMSE (minutes) |
|---|---|
| Baseline (mean per best-of format) | ≈ 34.95 |
| H2O AutoML (30 s training) | ≈ 33.9 |

Improvement: about **1 minute (≈ 3 %)**. AutoML results vary slightly between runs because training is limited by time.

---

## Tests

The tests are written with **pytest** in the *Given / When / Then* style and can be run with:

```bash
make test
```

They cover four types of tests:

| Type | Purpose | Examples |
|---|---|---|
| **Unit tests** | Check single functions with small, hand-made data | month conversion, name fixing, blinding of players, missing value handling, encoding, baseline RMSE calculation, improvement calculation |
| **Integration tests** | Check that the steps work together on the real dataset | columns are dropped correctly, no missing values after preprocessing, 80/20 split, no duplicate matches, valid RMSE values |
| **Regression tests** | Make sure results do not change unexpectedly | baseline RMSE stays the same, AutoML RMSE stays within ±10 % |
| **Model tests** | Check that results make sense from a domain perspective | no data leakage in the features, plausible ages, best-of-5 matches take longer than best-of-3, the baseline beats a constant prediction, AutoML beats the baseline |

> **Note:** The modeling script trains the H2O model when it is imported. Running the tests therefore takes about as long as one training run (around 1 minute with 30 seconds of training).
