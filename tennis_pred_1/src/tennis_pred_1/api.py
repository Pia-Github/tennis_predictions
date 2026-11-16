from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tennis_pred_1.Tennis_time_predictions_predict import predict_duration

app = FastAPI(title="Tennis match duration prediction")


class Match(BaseModel):
    player_1: str
    player_2: str
    tourney_name: str
    draw_size: int
    best_of: int
    t_round: str
    year: int
    surface: str
    tourney_level: str
    tourney_id: str
    tourney_date: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "player_1": "Bernard Tomic",
                "player_2": "Mattia Bellucci",
                "tourney_name": "San Jose",
                "draw_size": 32,
                "best_of": 3,
                "t_round": "QF",
                "year": 2026,
                "surface": "Hard",
                "tourney_level": "C",
                "tourney_id": "2024-9900",
                "tourney_date": "2026-10-03",
            }]
        }
    }


@app.post("/predict")
def predict(match: Match):
    try:
        return predict_duration(**match.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))