# base image: Linux with Python 3.11
FROM python:3.11-slim

# install Java (needed by H2O)
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# working directory inside the container
WORKDIR /app

# copy the project into the container
COPY . .

# install dependencies (without dev packages like pytest)
RUN uv sync --frozen --no-dev

# port of the API
EXPOSE 8000

# start the API
CMD ["uv", "run", "--no-dev", "fastapi", "run", "src/tennis_pred_1/api.py", "--host", "0.0.0.0", "--port", "8000"]

