import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML

tennis_df = pd.read_csv("ATP_tennis_group_project.csv")
tennis_df = tennis_df.drop(columns=["Unnamed: 0"])

tennis_df = pd.read_csv("ATP_tennis_group_project.csv")

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

tennis_df = tennis_df.drop(columns=cols_to_drop)

tennis_df["month"] = pd.to_datetime(tennis_df["tourney_date"]).dt.month
tennis_df["tourney_month_sin"] = np.sin(2 * np.pi * tennis_df["month"] / 12)
tennis_df["tourney_month_cos"] = np.cos(2 * np.pi * tennis_df["month"] / 12)
tennis_df = tennis_df.drop(columns=["tourney_date", "month"])
tennis_df.head()

tennis_df = tennis_df.replace("Edouard Roger-Vasselin", "Edouard Roger Vasselin")

mask = np.random.rand(len(tennis_df)) > 0.5

winner_cols = ['winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
               'winner_rank', 'winner_rank_points']
loser_cols  = ['loser_name',  'loser_hand',  'loser_ht',  'loser_ioc',  'loser_age',
               'loser_rank',  'loser_rank_points']
new_cols_1  = ['name_1', 'hand_1', 'ht_1', 'ioc_1', 'age_1', 'rank_1', 'rank_points_1']
new_cols_2  = ['name_2', 'hand_2', 'ht_2', 'ioc_2', 'age_2', 'rank_2', 'rank_points_2']

for w, l, c1, c2 in zip(winner_cols, loser_cols, new_cols_1, new_cols_2):
    tennis_df[c1] = np.where(mask, tennis_df[w], tennis_df[l])
    tennis_df[c2] = np.where(mask, tennis_df[l], tennis_df[w])

tennis_df.drop(columns=winner_cols + loser_cols, inplace=True)

long_df = tennis_df.copy()

long_df = long_df.drop_duplicates()

long_df = long_df.dropna(subset=["minutes"])
long_df = long_df[long_df["minutes"] > 0]

long_df["rank_1"] = long_df["rank_1"].fillna(9999)
long_df["rank_2"] = long_df["rank_2"].fillna(9999)
long_df["rank_points_1"] = long_df["rank_points_1"].fillna(0)
long_df["rank_points_2"] = long_df["rank_points_2"].fillna(0)


imputer = IterativeImputer(random_state=42)

long_df[["age_1", "ht_1", "age_2", "ht_2"]] = imputer.fit_transform(
    long_df[["age_1", "ht_1", "age_2", "ht_2"]]
)

cat_cols = ["tourney_id", "tourney_name", "name_1", "name_2", "ioc_1", "ioc_2"]

le = LabelEncoder()
for col in cat_cols:
    long_df[col] = le.fit_transform(long_df[col].astype(str))

long_df = pd.get_dummies(long_df, columns=["tourney_level", "surface", "hand_1", "hand_2"])
bool_cols = long_df.select_dtypes(include="bool").columns
long_df[bool_cols] = long_df[bool_cols].astype(int)

round_order = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    "RR": 3,
    "ER": 1,
    "BR": 6
}
long_df["round"] = long_df["round"].map(round_order)

X = long_df.drop(columns=["minutes"])
y = long_df["minutes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv("X_train")
X_test.to_csv("X_test")
y_train.to_csv("y_train")
y_test.to_csv("y_test")

print(X_test.isnull().sum()[X_test.isnull().sum() > 0])
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])
print(y_train.isnull().sum()[y_train.isnull().sum() > 0])
print(y_test.isnull().sum()[y_test.isnull().sum() > 0])


h2o.init()

train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test  = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

aml = H2OAutoML(max_runtime_secs=600, seed=42)
aml.train(y="minutes", training_frame=train)

print(aml.leaderboard)

for model_id in aml.leaderboard['model_id'].as_data_frame()['model_id']:
    if 'Stacked' not in model_id:
        best_single = h2o.get_model(model_id)
        break

varimp = best_single.varimp(use_pandas=True)
print(varimp.head(20))

rename_map = {
    'best_of': 'num sets',
    'draw_size': 'total participants',
    'tourney_level_G': 'Grand Slam',
    'tourney_id': 'tourney',
    'year': 'year',
    'rank_points_1': 'rank points 1',
    'round': 'round',
    'rank_points_2': 'rank points 2',
    'name_1': 'player 1',
    'name_2': 'player 2'
}

top10 = varimp.head(10).sort_values('percentage', ascending=True)
top10 = top10.copy()
top10['variable'] = top10['variable'].map(rename_map).fillna(top10['variable'])

baseline = long_df.groupby('best_of')['minutes'].mean()
print(baseline)

def predict_duration(best_of):
    return baseline[best_of]

y_pred = X_test['best_of'].map(baseline)
rmse_baseline = np.sqrt(((y_test - y_pred) ** 2).mean())
print(f"\nBaseline RMSE: {rmse_baseline:.2f}")

baseline = long_df.groupby('best_of')['minutes'].mean()
y_pred_baseline = X_test['best_of'].map(baseline)
rmse_baseline = np.sqrt(((y_test - y_pred_baseline) ** 2).mean())

rmse_automl = aml.leaderboard[0, 'rmse']

improvement = rmse_baseline - rmse_automl
improvement_pct = (improvement / rmse_baseline) * 100

print(f"Baseline RMSE (nur best_of): {rmse_baseline:.2f}")
print(f"AutoML RMSE:                 {rmse_automl:.2f}")
print(f"Improvement:                {improvement:.2f} minutes ({improvement_pct:.1f}%)")