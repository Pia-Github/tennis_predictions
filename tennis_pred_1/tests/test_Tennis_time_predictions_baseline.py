# Unit Testing: Does each module/function is actually provide the output it claims it does?
# Integration Testing: Do modules work nicely together?
# Regression Testing: ensure the bugs that you already detected and fixed are not re-appearing after a code change
# Model Testing: testing specific model behaviors (e.g. is better than other model?)
# Style: given - when - then

# "assert" for what should be tested
# temporary files can be useful


import numpy as np
import pandas as pd
import pytest

# When imporing, script runs completely
from tennis_pred_1 import Tennis_time_predictions_baseline as baseline

# ---------------------------------------------------------------
# Unit tests: baseline_model with small, self-built data
# ---------------------------------------------------------------

def test_baseline_model_returns_expected_rmse(monkeypatch):
    # Given: training data with known group means (best_of 3 -> 100, best_of 5 -> 200)
    train_df = pd.DataFrame({
        "best_of": [3, 3, 5, 5],
        "minutes": [90, 110, 180, 220],
    })
    # and test data that is 10 minutes away from the group mean in each case
    monkeypatch.setattr(baseline, "X_test", pd.DataFrame({"best_of": [3, 5]}))
    monkeypatch.setattr(baseline, "y_test", pd.Series([110, 190]))

    # When
    rmse = baseline.baseline_model(train_df)

    # Then
    assert rmse == pytest.approx(10.0)


def test_baseline_model_perfect_prediction_gives_zero(monkeypatch):
    # Given: test values that exactly match the group means
    train_df = pd.DataFrame({"best_of": [3, 5], "minutes": [100, 200]})
    monkeypatch.setattr(baseline, "X_test", pd.DataFrame({"best_of": [3, 5, 3]}))
    monkeypatch.setattr(baseline, "y_test", pd.Series([100, 200, 100]))

    # When
    rmse = baseline.baseline_model(train_df)

    # Then
    assert rmse == pytest.approx(0.0)


# ---------------------------------------------------------------
# Integration tests: CSV + Preprocessing + Split + Baseline
# ---------------------------------------------------------------

def test_pipeline_produces_valid_rmse():
    # Given: the script has loaded the real CSV and run the full pipeline on import
    # When
    rmse = baseline.rmse_baseline

    # Then
    assert isinstance(rmse, float)
    assert not np.isnan(rmse)
    assert rmse > 0


def test_split_output_fits_baseline_input():
    # Given: the test data created by preprocessing_splitting
    X_test, y_test = baseline.X_test, baseline.y_test

    # When: the group means are mapped onto the test data
    group_means = baseline.tennis_df.groupby("best_of")["minutes"].mean()
    y_pred = X_test["best_of"].map(group_means)

    # Then: every test match gets a prediction
    assert "best_of" in X_test.columns
    assert len(X_test) == len(y_test)
    assert y_pred.notna().all()


# ---------------------------------------------------------------
# Regression test: Results stay the same
# ---------------------------------------------------------------

def test_baseline_rmse_unchanged():
    # Given: the RMSE measured on 2026-11-08
    expected_rmse = 34.9479887882812

    # When
    rmse = baseline.rmse_baseline

    # Then
    assert rmse == pytest.approx(expected_rmse, abs=0.001)


# ---------------------------------------------------------------
# Model tests: does the baseline act sensful (contentwise)?
# ---------------------------------------------------------------

def test_best_of_5_predicts_longer_than_best_of_3():
    # Given: the real dataset
    df = baseline.tennis_df

    # When
    group_means = df.groupby("best_of")["minutes"].mean()

    # Then
    assert group_means[5] > group_means[3]


def test_baseline_beats_constant_mean_prediction():
    # Given: a naive model that always predicts the mean training duration
    constant_pred = baseline.y_train.mean()
    rmse_constant = np.sqrt(((baseline.y_test - constant_pred) ** 2).mean())

    # When
    rmse = baseline.rmse_baseline

    # Then
    assert rmse < rmse_constant