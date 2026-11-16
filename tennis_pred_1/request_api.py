import requests

url = "http://127.0.0.1:8000/predict"
match_data = {
    "player_1": "Bernard Tomic",
    "player_2": "Mattia Bellucci",
    "tourney_name": "Jingshan",
    "draw_size": 32,
    "best_of": 3,
    "t_round": "QF",
    "year": 2026,
    "surface": "Hard",
    "tourney_level": "C",
    "tourney_id": "2025-1024",
    "tourney_date": "2026-10-03",
}

response = requests.post(url, json=match_data).json()
print(response)
assert "model_prediction" in response
assert "baseline_prediction" in response