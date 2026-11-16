import os
import h2o
from tennis_pred_1.Upcoming_match_refactored import predict_match, tennis_df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "automl_leader")

# load model (once, when the script is imported)
h2o.init()
model = h2o.load_model(MODEL_PATH)

# baseline: mean duration of all matches per best_of
baseline = tennis_df.groupby("best_of")["minutes"].mean()


def predict_duration(player_1, player_2, tourney_name, draw_size, best_of,
                     t_round, year, surface, tourney_level, tourney_id, tourney_date):
    """
    Predicts the duration of an upcoming match with the H2O model and the baseline.

    Returns
    -------
    dict
        Predicted duration in minutes from the model and from the baseline.
    """
    # preprocess the match
    predict_df = predict_match(
        tennis_df, player_1, player_2, tourney_name, draw_size, best_of,
        t_round, year, surface, tourney_level, tourney_id, tourney_date)

    # convert predict_df to H2O Frame
    predict_df = predict_df.drop(columns=["minutes"])
    predict_h2o = h2o.H2OFrame(predict_df)

    # prediction
    pred = model.predict(predict_h2o)
    model_pred = pred.as_data_frame()["predict"][0]
    baseline_pred = baseline[best_of]

    return {
        "model_prediction": float(model_pred),
        "baseline_prediction": float(baseline_pred),
    }


if __name__ == "__main__":
    result = predict_duration(
        "Bernard Tomic", "Mattia Bellucci", "San Jose", 32, 3, "QF",
        2026, "Hard", "C", "2024-9900", "2026-10-03")
    print(result)