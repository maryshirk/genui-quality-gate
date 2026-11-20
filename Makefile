.PHONY: setup install-hooks lint format typecheck test test-integration test-containers check run worker source up down logs build smoke ml-up train eda ml-smoke pipeline-data experiment

export APP_VERSION = $(shell python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
export BUILD_SHA = $(shell git rev-parse HEAD)

setup:
	uv sync --locked --group ml
	$(MAKE) install-hooks

install-hooks:
	uv run --locked pre-commit install --install-hooks

lint:
	uv run --locked ruff check .

format:
	uv run --locked ruff format .

typecheck:
	uv run --locked --group ml mypy

test:
	uv run --locked --group ml pytest --cov --cov-report=term-missing --cov-report=xml --cov-report=html

test-integration:
	uv run --locked --group ml pytest -m integration

test-containers:
	RUN_CONTAINER_TESTS=1 uv run --locked --group ml pytest -m containers

check: lint typecheck
	uv run --locked ruff format --check .
	uv run --locked pre-commit run --all-files
	$(MAKE) test

run:
	GENUI_BUILD_SHA=$(BUILD_SHA) uv run --locked --env-file .env uvicorn genui_quality_gate.main:create_app --factory --reload --no-access-log

worker:
	uv run --locked --env-file .env uvicorn genui_quality_gate.worker:create_worker_app --factory --port 8001 --no-access-log --timeout-graceful-shutdown 70

source:
	uv run --locked --env-file .env python -m genui_quality_gate.source $(INPUT) --checkpoint $(CHECKPOINT)

pipeline-data:
	uv run --locked --group ml python -m scripts.pipeline_experiment prepare

experiment:
	uv run --locked --group ml --env-file .env python -m scripts.pipeline_experiment run

up:
	docker compose up --build --wait --wait-timeout 180

ml-up:
	docker compose up --build --wait --wait-timeout 180 postgres mlflow

train:
	uv run --locked --group ml --env-file .env python -m genui_quality_gate.ml.train

eda:
	uv run --locked --group ml --env-file .env python -m scripts.run_eda

ml-smoke:
	uv run --locked --group ml --env-file .env python -m scripts.mlflow_smoke seed
	GENUI_MODEL_NAME=genui-pairwise-quality-smoke docker compose up --build --wait --wait-timeout 180 app worker
	GENUI_MODEL_NAME=genui-pairwise-quality-smoke uv run --locked --group ml --env-file .env python -m scripts.mlflow_smoke verify

down:
	docker compose down

logs:
	docker compose logs -f app

build:
	docker compose build

smoke:
	uv run --locked python scripts/container_smoke.py
