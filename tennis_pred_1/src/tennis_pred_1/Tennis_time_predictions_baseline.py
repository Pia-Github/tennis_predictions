# import libraries
from tennis_pred_1.Tennis_time_predictions_preprocessing import preprocessing_splitting
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loguru import logger
import os

#load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tennis_df = pd.read_csv(os.path.join(BASE_DIR, "ATP_tennis_group_project.csv"))
logger.info("dataframe loaded")

X_train, X_test, y_train, y_test = preprocessing_splitting(tennis_df)

# ## Baseline Model
def baseline_model(df):
    """
    Computes a naive baseline RMSE using mean match duration per 'best_of' category.

    Groups matches by the number of sets ('best_of') and predicts the mean duration
    for each group. Used as a lower-bound benchmark to compare against AutoML.

    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame (used to compute group means).

    Returns:
        float: RMSE of the baseline predictions on the test set.
    """
    baseline = df.groupby('best_of')['minutes'].mean()
    y_pred = X_test['best_of'].map(baseline)
    rmse_baseline = np.sqrt(((y_test - y_pred) ** 2).mean())
    logger.info("Baseline Model complete")
    return rmse_baseline

rmse_baseline = baseline_model(tennis_df)
print(rmse_baseline)