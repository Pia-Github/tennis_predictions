.PHONY: install baseline model predict api test

install:
	uv sync

baseline:
	uv run python -m tennis_pred_1.Tennis_time_predictions_baseline

model:
	uv run python -m tennis_pred_1.Tennis_time_predictions_modeling

predict:
	uv run python -m tennis_pred_1.Tennis_time_predictions_predict

api:
	uv run fastapi dev src/tennis_pred_1/api.py

test:
	uv run pytest