import os
import h2o
from tennis_pred_1.Upcoming_match_refactored import predict_match, tennis_df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "automl_leader")

# input: upcoming match
player_1 = "Bernard Tomic"
player_2 = "Mattia Bellucci"
tourney_name = "San Jose"
draw_size = 32
best_of = 3
t_round = "QF"
year = 2026
surface = "Hard"
tourney_level = "C"
tourney_id = "2024-9900"
tourney_date = "2026-10-03"

# preprocess the match
predict_df = predict_match(
    tennis_df, player_1, player_2, tourney_name, draw_size, best_of,
    t_round, year, surface, tourney_level, tourney_id, tourney_date)

# load model
h2o.init()
model = h2o.load_model(MODEL_PATH)

# convert predict_df to H2O Frame
predict_df = predict_df.drop(columns=["minutes"])   # neu
predict_h2o = h2o.H2OFrame(predict_df)

# prediction
pred = model.predict(predict_h2o)
#print(pred)
print(f"Model prediction: {pred} minutes")

# baseline prediction: mean duration of all matches with the same best_of
baseline = tennis_df.groupby("best_of")["minutes"].mean()
baseline_pred = baseline[best_of]
print(f"Baseline prediction: {baseline_pred:.1f} minutes")