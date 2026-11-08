# When imporing, script runs completely
from tennis_pred_1 import Tennis_time_predictions_modeling as modeling

# ---------------------------------------------------------------
# Unit tests: Comparison baseline vs. modeling
# ---------------------------------------------------------------

#Unit compare baseline and modeling returns expected improvement
def test_comparison_baseline_modeling_returns_expected_impovement(monkeypatch):
    # Given: rmse_baseline is 40 und rmse_automl is 30
    monkeypatch.setattr(modeling, "rmse_baseline", 40)
    monkeypatch.setattr(modeling, "rmse_automl", 30)

    # When
    improvement = modeling.compare_trained_baseline(None)

    # Then
    assert improvement == "Improvement: 10.00 minutes (25.0%)"

# ---------------------------------------------------------------
# Integration tests: after import, rmse_automl is a positive number
# ---------------------------------------------------------------

def test_training_produces_positive_rmse_automl():
    # Given: the script has loaded the real CSV, run preprocessing and trained AutoML on import
    # When
    rmse_automl = modeling.rmse_automl

    # Then
    assert rmse_automl > 0

# ---------------------------------------------------------------
# Regression test: rmse_automl stays similar
# ---------------------------------------------------------------

def test_modeling_rmse_automl_similar():
    # Given: the RMSE measured on 2026-11-08
    expected_rmse = 33.95627463148948

    # When
    rmse_automl = modeling.rmse_automl

    # Then
    assert rmse_automl > expected_rmse - expected_rmse*0.1
    assert rmse_automl < expected_rmse + expected_rmse*0.1


# ---------------------------------------------------------------
# Model tests: automl is better than baseline (rmse_automl < rmse_baseline)
# ---------------------------------------------------------------

def test_automl_smaller_rmse_than_baseline():
    # Given: both models were evaluated on import

    # When
    rmse_automl = modeling.rmse_automl
    rmse_baseline = modeling.rmse_baseline

    # Then
    assert rmse_automl < rmse_baseline