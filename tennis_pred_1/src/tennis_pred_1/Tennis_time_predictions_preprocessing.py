# import libraries
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loguru import logger

#load dataset
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#tennis_df = pd.read_csv(os.path.join(BASE_DIR, "ATP_tennis_group_project.csv"))
#logger.info("dataframe loaded")

# Data Preprocessing
def drop_coulumns(df):
    """
    Drops irrelevant columns from the tennis dataset.

    Removes match statistics collected after the match (e.g. aces, break points),
    as well as ID and metadata columns that do not contribute to predicting match duration.

    Parameters:
        df (pd.DataFrame): Raw tennis DataFrame.

    Returns:
        pd.DataFrame: DataFrame with irrelevant columns removed.
    """
    cols_to_drop = [
        "Unnamed: 0", "winner_id", "loser_id",
        "score",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", 
        "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", 
        "l_bpFaced",
        "winner_seed", "loser_seed",
        "winner_entry", "loser_entry", "has_stats", "match_num"
    ]
    
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    
    return df

def date_to_month(df):
    """
    Converts tournament date to cyclical month features.

    Extracts the month from 'tourney_date' and encodes it as sine and cosine
    to preserve the cyclical nature of months (e.g. December and January are close).
    The original date and intermediate month columns are dropped afterwards.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'tourney_date' column.

    Returns:
        pd.DataFrame: DataFrame with 'tourney_month_sin' and 'tourney_month_cos' columns added.
    """
    df["month"] = pd.to_datetime(df["tourney_date"]).dt.month
    df["tourney_month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["tourney_month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df = df.drop(columns=["tourney_date", "month"])
    return df


def fix_names(df):
    """
    Fixes inconsistent player name formatting in the dataset.

    Standardizes known name variations to ensure consistent player identification
    across the dataset (e.g. for merging or grouping by player name).

    Parameters:
        df (pd.DataFrame): DataFrame containing player name columns.

    Returns:
        pd.DataFrame: DataFrame with corrected player names.
    """
    df = df.replace("Edouard Roger-Vasselin", "Edouard Roger Vasselin")
    return df


def blinding_winning_losing(df):
    """
    Anonymizes winner/loser labels by randomly assigning players to 'player_1' and 'player_2'.

    For each match, randomly decides whether the winner becomes player_1 or player_2,
    preventing the model from learning that player_1 always wins. This avoids
    data leakage and ensures the model cannot exploit label ordering.

    Parameters:
        df (pd.DataFrame): DataFrame containing winner_* and loser_* columns.

    Returns:
        pd.DataFrame: DataFrame with anonymized player columns (name_1/2, hand_1/2, etc.)
                      and original winner/loser columns removed.
    """
    mask = np.random.rand(len(df)) > 0.5

    winner_cols = ['winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
                   'winner_rank', 'winner_rank_points']
    loser_cols  = ['loser_name',  'loser_hand',  'loser_ht',  'loser_ioc',  'loser_age',
                   'loser_rank',  'loser_rank_points']
    new_cols_1  = ['name_1', 'hand_1', 'ht_1', 'ioc_1', 'age_1', 'rank_1', 'rank_points_1']
    new_cols_2  = ['name_2', 'hand_2', 'ht_2', 'ioc_2', 'age_2', 'rank_2', 'rank_points_2']

    for w, l, c1, c2 in zip(winner_cols, loser_cols, new_cols_1, new_cols_2):
        df[c1] = np.where(mask, df[w], df[l])
        df[c2] = np.where(mask, df[l], df[w])

    df.drop(columns=winner_cols + loser_cols, inplace=True)

    return df


def deduplicate(df):
    """
    Removes duplicate rows from the dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicate rows removed.
    """
    df = df.drop_duplicates()
    return df

def missing_data_handling(df):
    """
    Handles missing values in the dataset.

    - Removes rows where 'minutes' is missing or zero (invalid matches).
    - Fills missing rank with 9999 (unranked players) and rank_points with 0.
    - Imputes missing age and height values using MICE (Multiple Imputation by Chained Equations).

    Parameters:
        df (pd.DataFrame): DataFrame after blinding winner/loser columns.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    #delete rows without minutes
    df = df.dropna(subset=["minutes"])
    df = df[df["minutes"] > 0]

    #fill empty rank/rankpoints with 9999 or 0
    df["rank_1"] = df["rank_1"].fillna(9999)
    df["rank_2"] = df["rank_2"].fillna(9999)
    df["rank_points_1"] = df["rank_points_1"].fillna(0)
    df["rank_points_2"] = df["rank_points_2"].fillna(0)

    # impute age and ht with MICE
    imputer = IterativeImputer(random_state=42)
    df[["age_1", "ht_1", "age_2", "ht_2"]] = imputer.fit_transform(
        df[["age_1", "ht_1", "age_2", "ht_2"]])
    
    return df


def data_encoding(df):
    """
    Encodes categorical variables for model training.

    - Label Encoding: Applied to high-cardinality columns (tournament ID/name, player names, nationality).
    - One-Hot Encoding: Applied to low-cardinality nominal columns (surface, tournament level, handedness).
    - Ordinal Encoding: Applied to 'round' based on match progression order.

    Parameters:
        df (pd.DataFrame): DataFrame after missing value handling.

    Returns:
        pd.DataFrame: DataFrame with all categorical columns numerically encoded.
    """
    #label encoding
    cat_cols = ["tourney_id", "tourney_name", "name_1", "name_2", "ioc_1", "ioc_2"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    #one hot encoding
    df = pd.get_dummies(df, columns=["tourney_level", "surface", "hand_1", "hand_2"])
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # ordinal encoding
    round_order = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4,
        "QF": 5, "SF": 6, "F": 7,
        "RR": 3,
        "ER": 1,
        "BR": 6
    }
    df["round"] = df["round"].map(round_order)
    
    return df


# ## Preprocessing complete
def preprocessing(df):
    """
    Runs the full preprocessing pipeline on the raw tennis dataset.

    Applies the following steps in order:
    1. Drop irrelevant columns
    2. Convert tournament date to cyclical month features
    3. Fix inconsistent player names
    4. Blind winner/loser labels by random assignment to player_1/player_2
    5. Remove duplicate rows
    6. Handle missing values
    7. Encode categorical variables

    Parameters:
        df (pd.DataFrame): Raw tennis DataFrame.

    Returns:
        pd.DataFrame: Fully preprocessed DataFrame ready for model training.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    logger.info("Starting preprocessing")
    
    df = drop_coulumns(df)
    df = date_to_month(df)
    df = fix_names(df)
    df = blinding_winning_losing(df)
    df = deduplicate(df)
    df = missing_data_handling(df)
    df = data_encoding(df)
    logger.info("Preprocessing complete")
    return df

# Train Test Split
def splitting(df):
    """
    Splits the dataset into training and test sets.

    Separates features from the target variable 'minutes' and applies
    an 80/20 train-test split with a fixed random seed for reproducibility.

    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame containing a 'minutes' column.

    Returns:
        tuple: X_train, X_test, y_train, y_test as pandas DataFrames/Series.
    """
    X = df.drop(columns=["minutes"])
    y = df["minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Train test split complete")
    return X_train, X_test, y_train, y_test


# combine preprocessing and splitting
def preprocessing_splitting(df):
    """
    Runs the full preprocessing pipeline and splits the data into train/test sets.

    Parameters:
        df (pd.DataFrame): Raw tennis DataFrame.

    Returns:
        tuple: X_train, X_test, y_train, y_test as pandas DataFrames/Series.
    """
    df = preprocessing(df)
    X_train, X_test, y_train, y_test = splitting(df)
    return X_train, X_test, y_train, y_test