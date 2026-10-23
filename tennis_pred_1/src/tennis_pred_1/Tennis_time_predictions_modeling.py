# import libraries
from tennis_pred_1.Tennis_time_predictions_preprocessing import preprocessing, preprocessing_splitting
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loguru import logger
import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home"
import h2o
from h2o.automl import H2OAutoML
h2o.init()

#load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tennis_df = pd.read_csv(os.path.join(BASE_DIR, "ATP_tennis_group_project.csv"))
logger.info("dataframe loaded")

X_train, X_test, y_train, y_test = preprocessing_splitting(tennis_df)

# Train Model
def train_model(X_train, y_train, X_test, y_test):
    """
    Trains an H2O AutoML model and evaluates it on the test set.

    Converts the input data to H2O frames, runs AutoML for 30 seconds,
    and returns the RMSE of the best model evaluated on the test set.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target ('minutes').
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target ('minutes').

    Returns:
        float: RMSE of the best AutoML model on the test set.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training data is empty")

    logger.info("Starting Training of model")

    train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    test  = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

    aml = H2OAutoML(max_runtime_secs=30, seed=42)
    aml.train(y="minutes", training_frame=train)
    
    perf = aml.leader.model_performance(test)
    rmse_automl = perf.rmse()

    logger.info("Model complete")
    return rmse_automl

rmse_automl = train_model(X_train, y_train, X_test, y_test)
print(rmse_automl)


# Improvement Trained Model vs. Baseline Model
from tennis_pred_1.Tennis_time_predictions_baseline import rmse_baseline

def compare_trained_baseline(df):
    """
    Compares the AutoML model RMSE against the baseline RMSE.

    Calculates the absolute and relative improvement of the AutoML model
    over the naive baseline prediction.

    Parameters:
        df (pd.DataFrame): Not used directly; included for pipeline consistency.

    Returns:
        str: Formatted string showing improvement in minutes and as a percentage.
    """
    improvement = rmse_baseline - rmse_automl
    improvement_pct = (improvement / rmse_baseline) * 100

    logger.info("Comparison complete")
    return f"Improvement: {improvement:.2f} minutes ({improvement_pct:.1f}%)"

print(compare_trained_baseline(tennis_df))