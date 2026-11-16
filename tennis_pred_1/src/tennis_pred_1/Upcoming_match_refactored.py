# Get packages and read data
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tennis_df = pd.read_csv(os.path.join(BASE_DIR, "ATP_tennis_group_project.csv"))
tennis_df = tennis_df.drop(columns=["Unnamed: 0"])

from tennis_pred_1 import Tennis_time_predictions_preprocessing as training_preprocessing

# same preprocessing steps as for the model training -> preprocessed data and legend
df_train = tennis_df.copy()
df_train = training_preprocessing.drop_coulumns(df_train)
df_train = training_preprocessing.date_to_month(df_train)
df_train = training_preprocessing.fix_names(df_train)
df_train = training_preprocessing.blinding_winning_losing(df_train)
df_train = training_preprocessing.deduplicate(df_train)
df_train = training_preprocessing.missing_data_handling(df_train)
tennis_df_preprocessed, legend = training_preprocessing.data_encoding(df_train)


# # Get data from players
PLAYER_ATTRS = ["name", "hand", "ht", "ioc", "age", "rank", "rank_points"]

def get_latest_player_info(df, player):
    """
    Returns the attributes of a player from their most recent match in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        ATP tennis dataset.
    player : str
        Full name of the player (e.g. 'Bernard Tomic').

    Returns
    -------
    dict
        Player attributes (name, hand, ht, ioc, age, rank, rank_points).
    """
    matches = df[(df["winner_name"] == player) | (df["loser_name"] == player)]
    if matches.empty:
        raise ValueError(f"Player '{player}' not found in the dataset")

    latest = matches.loc[pd.to_datetime(matches["tourney_date"]).idxmax()]
    role = "winner" if latest["winner_name"] == player else "loser"
    return {attr: latest[f"{role}_{attr}"] for attr in PLAYER_ATTRS}


# ## Get data from tournament
def build_match_row(df, player_1, player_2, tourney_values):
    """
    Builds one raw match row: player_1 vs. player_2 with tournament data.

    Player 1 is stored in the winner_* columns and player 2 in the loser_* columns,
    so the row has the same format as the raw dataset. Who actually wins is unknown;
    blinding in the preprocessing randomizes the order anyway.

    Parameters
    ----------
    df : pd.DataFrame
        ATP tennis dataset.
    player_1, player_2 : str
        Full names of the two players.
    tourney_values : dict
        Tournament data (column name -> value).

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly one row.
    """
    p1 = get_latest_player_info(df, player_1)
    p2 = get_latest_player_info(df, player_2)

    row = {col: np.nan for col in df.columns}
    for attr in PLAYER_ATTRS:
        row[f"winner_{attr}"] = p1[attr]
        row[f"loser_{attr}"] = p2[attr]
    row.update(tourney_values)

    return pd.DataFrame([row])


# ## Data Preprocessing
# drop columns
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

# extract month from tourney date
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

# get players into one row
def order_players(df, legend):
    """
    Assigns the two players to player 1 and player 2 so that both are known in the legend.

    Instead of a random order (as in blinding_winning_losing), the order is chosen
    in which player 1 exists in the legend for 'name_1' and player 2 for 'name_2'.
    """
    names_1 = set(legend.loc[legend["column"] == "name_1", "value"])
    names_2 = set(legend.loc[legend["column"] == "name_2", "value"])

    winner = df["winner_name"].iloc[0]
    loser = df["loser_name"].iloc[0]

    if winner in names_1 and loser in names_2:
        keep_order = True       # winner_* -> player 1, loser_* -> player 2
    elif loser in names_1 and winner in names_2:
        keep_order = False      # swapped
    else:
        keep_order = True       # no order fits completely, the encoding will warn

    winner_cols = ['winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
                   'winner_rank', 'winner_rank_points']
    loser_cols  = ['loser_name',  'loser_hand',  'loser_ht',  'loser_ioc',  'loser_age',
                   'loser_rank',  'loser_rank_points']
    new_cols_1  = ['name_1', 'hand_1', 'ht_1', 'ioc_1', 'age_1', 'rank_1', 'rank_points_1']
    new_cols_2  = ['name_2', 'hand_2', 'ht_2', 'ioc_2', 'age_2', 'rank_2', 'rank_points_2']

    for w, l, c1, c2 in zip(winner_cols, loser_cols, new_cols_1, new_cols_2):
        if keep_order:
            df[c1] = df[w]
            df[c2] = df[l]
        else:
            df[c1] = df[l]
            df[c2] = df[w]

    df = df.drop(columns=winner_cols + loser_cols)
    return df

# remove duplicates
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

# fill missing data
FILL_ATTRS = ["rank", "rank_points", "age", "ht"]
def missing_data_handling(df, df_preprocessed, legend):
    """
    Fills missing rank, rank points, age and height of both players
    with the values from their most recent match in df_preprocessed.
    """
    for pos in [1, 2]:
        name = df[f"name_{pos}"].iloc[0]
        missing = [attr for attr in FILL_ATTRS if pd.isna(df[f"{attr}_{pos}"].iloc[0])]
        if not missing:
            continue

        # numbers of the player in name_1 and name_2 (from the legend)
        code_1 = legend.loc[(legend["column"] == "name_1") & (legend["value"] == name), "code"].astype(int)
        code_2 = legend.loc[(legend["column"] == "name_2") & (legend["value"] == name), "code"].astype(int)

        # all matches where the player was player 1 or player 2
        rows_1 = df_preprocessed[df_preprocessed["name_1"].isin(code_1)]
        rows_2 = df_preprocessed[df_preprocessed["name_2"].isin(code_2)]

        # most recent match = last row (and whether the player was player 1 or 2 there)
        candidates = []
        if not rows_1.empty:
            candidates.append((rows_1.index.max(), 1))
        if not rows_2.empty:
            candidates.append((rows_2.index.max(), 2))
        if not candidates:
            warnings.warn(f"No data for '{name}' in tennis_df_preprocessed; still missing: {', '.join(missing)}")
            continue

        row_idx, source_pos = max(candidates)
        for attr in missing:
            df[f"{attr}_{pos}"] = df_preprocessed.at[row_idx, f"{attr}_{source_pos}"]

    return df

# data encoding
def data_encoding(df, legend):
    """
    Encodes categorical variables of the upcoming match for model input.

    - Label Encoding: Uses the legend from the training data for tournament ID/name,
      player names and nationality. Values that do not exist in the legend
      are encoded as -1 and a warning is shown.
    - One-Hot Encoding: Applied to tournament level, surface and handedness.
    - Ordinal Encoding: Applied to 'round' based on match progression order.

    Parameters
    ----------
    df : pd.DataFrame
        Match data after ordering the players and handling missing values.
    legend : pd.DataFrame
        Encoding legend from the training data with the columns
        'column' (original column), 'value' (original value) and 'code' (encoded number).

    Returns
    -------
    pd.DataFrame
        DataFrame with all categorical columns numerically encoded.
    """
    #label encoding with legend
    cat_cols = ["tourney_id", "tourney_name", "name_1", "name_2", "ioc_1", "ioc_2"]
    for col in cat_cols:
        col_legend = legend[legend["column"] == col]
        mapping = dict(zip(col_legend["value"], col_legend["code"]))

        for value in df[col].astype(str):
            if value not in mapping:
                warnings.warn(f"'{value}' in column '{col}' not in legend -> -1")

        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

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

# #### Preprocessing complete
def preprocessing(df):
    """
    Runs the full preprocessing pipeline on the raw tennis dataset.
    ...
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    df = drop_coulumns(df)
    df = date_to_month(df)
    df = order_players(df, legend)
    df = deduplicate(df)
    df = missing_data_handling(df, tennis_df_preprocessed, legend)
    df = data_encoding(df, legend)
    return df


# ## Full pipeline
def predict_match(df, player_1, player_2, tourney_name, draw_size, best_of,
                  t_round, year, surface, tourney_level, tourney_id, tourney_date):
    """
    Full pipeline: builds the match row and preprocesses it for model input.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with one row (player_1 vs. player_2).
    """
    tourney_values = {
        "tourney_name": tourney_name,
        "draw_size": draw_size,
        "best_of": best_of,
        "round": t_round,
        "year": year,
        "surface": surface,
        "tourney_level": tourney_level,
        "tourney_id": tourney_id,
        "tourney_date": tourney_date,
    }

    predict_df = build_match_row(df, player_1, player_2, tourney_values)
    predict_df = preprocessing(predict_df)
    return predict_df