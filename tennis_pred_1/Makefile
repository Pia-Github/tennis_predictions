.PHONY: install baseline model test

install:
	uv sync

baseline:
	uv run python -m tennis_pred_1.Tennis_time_predictions_baseline

model:
	uv run python -m tennis_pred_1.Tennis_time_predictions_modeling

test:
	uv run pytest