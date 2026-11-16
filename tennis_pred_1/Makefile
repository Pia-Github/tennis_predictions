.PHONY: install baseline model predict api request test docker-build docker-run

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

request:
	uv run python request_api.py

test:
	uv run pytest

docker-build:
	docker build -t tennis-pred .

docker-run:
	docker run -p 8000:8000 tennis-pred