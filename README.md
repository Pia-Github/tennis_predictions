# Tennis Match Duration Prediction

This project predicts **how long an ATP tennis match will last (in minutes)** using information that is available *before* the match starts (tournament, surface, round, best-of format, player rankings, age, height, etc.).

It compares two approaches:

1. **Baseline model:** predicts the average match duration depending only on whether the match is played best-of-3 or best-of-5.
2. **Machine learning model:** an H2O AutoML model trained on all preprocessed features.

The goal is to see how much a machine learning model improves on the simple baseline.

Both models can be used to predict the duration of an **upcoming match** – via a script, a **FastAPI** web API, a **Docker** container or the deployed API on **AWS**.

---

## Project structure

```
tennis_predictions/
├── README.md
├── .gitignore
└── tennis_pred_1/
    ├── Makefile                      # shortcuts for install, run, test and docker
    ├── Dockerfile                    # builds the Docker image of the API
    ├── .dockerignore                 # files that are not copied into the image
    ├── pyproject.toml                # project definition and dependencies
    ├── uv.lock                       # exact dependency versions (managed by uv)
    ├── .python-version
    ├── request_api.py                # sends a test request to the running API
    ├── models/
    │   └── automl_leader             # trained H2O model (created by `make model`)
    ├── notebooks/
    │   └── Tennis_data_master_table.ipynb
    ├── src/
    │   └── tennis_pred_1/
    │       ├── __init__.py
    │       ├── ATP_tennis_group_project.csv               # raw dataset
    │       ├── Tennis_time_predictions_preprocessing.py   # data preparation
    │       ├── Tennis_time_predictions_baseline.py        # baseline model
    │       ├── Tennis_time_predictions_modeling.py        # H2O AutoML model + comparison
    │       ├── Upcoming_match_refactored.py               # prepares an upcoming match for prediction
    │       ├── Tennis_time_predictions_predict.py         # predicts an upcoming match (model + baseline)
    │       ├── api.py                                     # FastAPI app
    │       └── ping.py                                    # minimal FastAPI example
    └── tests/
        ├── test_Tennis_time_predictions_preprocessing.py
        ├── test_Tennis_time_predictions_baseline.py
        └── test_Tennis_time_predictions_modeling.py
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
- **Java** (required by H2O; it must be installed and available on the `PATH` or via `JAVA_HOME`)
- **Docker** (only for running the API in a container)
- **AWS CLI** (only for deploying to AWS)

---

## Getting started

All commands are run from the `tennis_pred_1` folder (where the `Makefile` and `pyproject.toml` are located):

```bash
cd tennis_pred_1
make install
make model
```

`make install` runs `uv sync` and installs all dependencies in the exact versions defined in `uv.lock`, including the project package itself and the development dependencies `pytest` and `requests`.

`make model` trains the model and saves it to `models/automl_leader`. This is required before predictions, the API or the Docker image can be used.

---

## Usage

| Command | What it does |
|---|---|
| `make install` | Installs all dependencies (`uv sync`) |
| `make baseline` | Runs the baseline model and prints its RMSE |
| `make model` | Trains and saves the H2O AutoML model, prints its RMSE and the improvement over the baseline |
| `make predict` | Predicts the duration of an example upcoming match (model and baseline) |
| `make api` | Starts the FastAPI app locally (development mode) |
| `make request` | Sends a test request to the running API (`make api` must be running in another terminal) |
| `make test` | Runs all tests with pytest |
| `make docker-build` | Builds the Docker image `tennis-pred` |
| `make docker-run` | Starts the API in a Docker container on port 8000 |

`make model` also runs the baseline (the modeling script imports the baseline result for the comparison), so it covers the complete training pipeline.

### Training time (environment variable)

The AutoML training time is read from the environment variable `MAX_RUNTIME_SECS`. If it is not set, 30 seconds are used.

```bash
make model                          # 30 seconds (default)
MAX_RUNTIME_SECS=600 make model     # 10 minutes, only for this command
```

---

## Pipeline

### 1. Preprocessing (`Tennis_time_predictions_preprocessing.py`)

This file only defines functions; it is used by the other scripts. The function `preprocessing_splitting(df)` runs all steps in this order:

1. **Drop columns:** removes IDs, metadata and all match statistics that are only known *after* the match (e.g. aces, break points, score) to avoid data leakage.
2. **Date → month:** converts `tourney_date` into the month and encodes it as sine and cosine, so that December and January are treated as close to each other.
3. **Fix names:** unifies known spelling variants of player names.
4. **Blind winner/loser:** randomly assigns winner and loser to `player_1` and `player_2`, so the model cannot learn that player 1 is always the winner. A fixed seed (`42`) makes the assignment identical in every run, so the encoding is always the same as during model training.
5. **Remove duplicates**
6. **Handle missing values:**
   - removes matches with missing or 0 minutes,
   - missing rank → `9999` (unranked), missing rank points → `0`,
   - missing age and height are imputed with MICE (`IterativeImputer`).
7. **Encoding:**
   - label encoding for tournament ID/name, player names and nationality,
   - one-hot encoding for tournament level, surface and handedness,
   - ordinal encoding for the round (R128 → 1 … F → 7).

   `data_encoding` additionally returns a **legend** that lists every label-encoded value and its number. The legend is used to encode upcoming matches the same way as the training data.
8. **Train/test split:** 80 % training, 20 % test (`random_state=42`), target variable `minutes`.

### 2. Baseline model (`Tennis_time_predictions_baseline.py`)

Calculates the mean match duration per `best_of` value (3 or 5 sets) and uses it as the prediction. The result is the RMSE on the test set.

### 3. Modeling (`Tennis_time_predictions_modeling.py`)

Trains an **H2O AutoML** regression model on the training data, saves the best model to `models/automl_leader` and evaluates it on the test set. Afterwards it compares the RMSE of the AutoML model with the baseline RMSE and prints the improvement in minutes and percent.

The training time is set via the environment variable `MAX_RUNTIME_SECS` (see [Training time](#training-time-environment-variable)).

---

## Predicting an upcoming match

### Preparing the match (`Upcoming_match_refactored.py`)

Prepares one upcoming match so that it has the same format as the training data:

1. **Player data:** looks up both players in the raw dataset and takes their attributes (hand, height, nationality, age, rank, rank points) from their **most recent match**. If a player is not in the dataset, a `ValueError` is raised.
2. **Tournament data:** adds the tournament information given by the user.
3. **Player order:** instead of a random order, the players are assigned to `player_1` and `player_2` so that both are known in the legend.
4. **Missing values:** missing rank, rank points, age and height are filled with the values from the player's most recent match in the preprocessed training data.
5. **Encoding:** label encoding with the legend from the training data. Values that did not exist in the training data (e.g. a new tournament) are encoded as `-1` and a warning is shown.

The preprocessed training data and the legend are created directly in code with the same preprocessing steps as for the model training (no CSV files needed).

### Prediction (`Tennis_time_predictions_predict.py`)

The function `predict_duration(...)` takes the match information, prepares the match and returns both predictions:

```python
{"model_prediction": 87.3, "baseline_prediction": 98.4}
```

- **Model prediction:** the saved H2O model (`models/automl_leader`) is loaded once when the script is imported.
- **Baseline prediction:** mean duration of all matches with the same `best_of`.

`make predict` runs the function for an example match.

---

## API (`api.py`)

The FastAPI app makes the prediction available as a web API.

| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Redirects to the documentation (`/docs`) |
| `/docs` | GET | Interactive documentation (Swagger UI) to try out the API |
| `/predict` | POST | Returns the model and baseline prediction for a match |

Example request body for `/predict`:

```json
{
  "player_1": "Bernard Tomic",
  "player_2": "Mattia Bellucci",
  "tourney_name": "Jingshan",
  "draw_size": 32,
  "best_of": 3,
  "t_round": "QF",
  "year": 2026,
  "surface": "Hard",
  "tourney_level": "C",
  "tourney_id": "2025-1024",
  "tourney_date": "2026-10-03"
}
```

The input is validated automatically. If a player is not found in the dataset, the API returns status `400` with an error message.

**Run locally:**

```bash
make api
```

Then open `http://127.0.0.1:8000` (in GitHub Codespaces: open the forwarded address of port 8000 in the **Ports** tab).

**Test request:** `request_api.py` sends the example match to the running API and checks that both predictions are returned:

```bash
make request    # in a second terminal while `make api` is running
```

---

## Docker

The `Dockerfile` packages the API with everything it needs (Python 3.11, Java, uv, all dependencies, code, data and the trained model). `.dockerignore` excludes the local virtual environment and cache folders.

```bash
make model          # the trained model must exist before building
make docker-build   # build the image (again after every change to code, data or model)
make docker-run     # start the container, API available on port 8000
```

In the container, the API is started with `fastapi run` (production mode) instead of `fastapi dev`.

---

## Deployment on AWS

The Docker image is deployed on AWS with **Amazon ECR** (image storage) and **Amazon ECS Express Mode** (runs the container and creates a public HTTPS URL). The deployment was done with a new AWS account on the **Free plan** in the region **eu-north-1 (Stockholm)**.

### 1. Connect the AWS CLI to the account

```bash
aws login --remote
```

### 2. Upload the image to ECR

```bash
aws ecr create-repository --repository-name tennis-pred --region eu-north-1

aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-north-1.amazonaws.com

docker tag tennis-pred:latest <account-id>.dkr.ecr.eu-north-1.amazonaws.com/tennis-pred:latest
docker push <account-id>.dkr.ecr.eu-north-1.amazonaws.com/tennis-pred:latest
```

`<account-id>` is the 12-digit AWS account ID. The repository only needs to be created once.

### 3. Create the service with ECS Express Mode

In the AWS Console: **Elastic Container Service** → **Create** → **Express mode**

- **Image:** `tennis-pred:latest` from ECR
- **Task execution role / infrastructure role:** create new roles
- **Container port:** `8000`
- **Health check path:** `/docs` (`/` only returns a redirect, not `200`; with a wrong path, AWS considers the container unhealthy and stops it again and again)

After a few minutes, the service shows an **application URL**. It redirects directly to the API documentation.

### Updating the deployment

After changes to the code, data or model:

1. `make docker-build`
2. Log in to ECR again and push the image (step 2)
3. In the ECS Console: **Service aktualisieren / Update service** → **Browse ECR images** → select `latest` again → **Update**

### Costs and clean-up

The running service uses Free Tier credits even when nobody uses it. On the Free plan, the account is closed after 6 months or when the credits are used up. When the API is no longer needed, delete the service in the ECS Console (**Delete service**).

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

The running API is not part of `make test`; it is checked separately with `make request`.

---

## Known limitations

This is a practice project. Some simplifications:

- **Player names** must be written exactly as in the dataset.
- **New tournaments** (not in the training data) are encoded as `-1`.
- **One-hot columns** of an upcoming match only contain the values of this match (e.g. only `surface_Hard`). H2O fills the missing one-hot columns with missing values instead of `0`, which can slightly distort the model prediction.
- The deployed API has **no authentication**; anyone with the URL can use it.
