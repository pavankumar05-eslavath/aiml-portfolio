# Convenience targets. Everything here is a thin wrapper over a command that also
# works standalone, so nothing is hidden behind the Makefile.
#
#   make help
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Use the project venv when present, otherwise whatever python is on PATH.
# CI installs into the runner's interpreter, so this must not hard-code .venv.
PYTHON  ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python)
PIP     ?= $(shell test -x .venv/bin/pip && echo .venv/bin/pip || echo pip)
BACKEND := backend

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------- setup
.PHONY: venv
venv: ## Create the virtualenv and install everything (CPU torch)
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r $(BACKEND)/requirements-dev.txt
	@echo "Now: cp .env.example .env && make sample && make index"

.PHONY: env
env: ## Create .env from the example if it does not exist
	@test -f .env || (cp .env.example .env && echo "created .env")

# -------------------------------------------------------------------- data
# `data` and `all` are the portfolio-wide entry points every project provides.
.PHONY: data
data: env sample ## Fetch the dataset images for the committed catalogue

.PHONY: all
all: data index evaluate ## Run every stage: fetch data, index, evaluate

.PHONY: sample
sample: ## Fetch images for the committed 1,000-product sample catalogue
	$(PYTHON) scripts/prepare_sample.py

.PHONY: dataset
dataset: ## Build a larger catalogue (SIZE=6000 OUT=data/eval)
	$(PYTHON) scripts/download_dataset.py --limit $(or $(SIZE),6000) --out $(or $(OUT),data/eval)

.PHONY: index
index: ## Load and index the sample catalogue
	$(PYTHON) scripts/index_catalog.py

.PHONY: reindex
reindex: ## Re-embed everything from scratch (after changing MODEL_NAME)
	$(PYTHON) scripts/index_catalog.py --force --recreate

# ------------------------------------------------------------------- serve
.PHONY: api
api: ## Run the API with autoreload
	$(PYTHON) -m uvicorn app.main:app --app-dir $(BACKEND) --reload --port 8000

.PHONY: ui
ui: ## Run the Vite dev server
	cd frontend && npm run dev

# ------------------------------------------------------------------- checks
.PHONY: test
test: ## Run the fast test suite
	cd $(BACKEND) && ../$(PYTHON) -m pytest

.PHONY: test-all
test-all: ## Run every test, including those that load the real model
	cd $(BACKEND) && RUN_SLOW_TESTS=1 ../$(PYTHON) -m pytest

.PHONY: cov
cov: ## Run tests with a coverage report
	cd $(BACKEND) && ../$(PYTHON) -m pytest --cov=app --cov-report=term-missing

# scripts/ lives outside backend/ but shares its ruff configuration, so it must be
# passed explicitly - otherwise it is silently never linted.
RUFF_CFG := --config $(BACKEND)/pyproject.toml

.PHONY: lint
lint: ## Lint and type-check the backend and scripts, type-check the frontend
	cd $(BACKEND) && ../$(PYTHON) -m ruff check . && ../$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check scripts/ $(RUFF_CFG)
	$(PYTHON) -m ruff format --check scripts/ $(RUFF_CFG)
	cd $(BACKEND) && ../$(PYTHON) -m mypy
	cd frontend && npx tsc -b && npx oxlint

.PHONY: format
format: ## Auto-format backend, scripts and frontend
	cd $(BACKEND) && ../$(PYTHON) -m ruff format . && ../$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format scripts/ $(RUFF_CFG)
	$(PYTHON) -m ruff check --fix scripts/ $(RUFF_CFG)
	cd frontend && npx prettier --write "src/**/*.{ts,tsx,css}" 2>/dev/null || true

# --------------------------------------------------------------- evaluation
.PHONY: evaluate
evaluate: ## Measure retrieval quality and write the report
	$(PYTHON) scripts/evaluate.py --per-query

.PHONY: experiments
experiments: ## Run the ranking parameter sweeps
	$(PYTHON) scripts/run_experiments.py

.PHONY: verify-model
verify-model: ## Measure whether the model's image/text spaces are aligned
	$(PYTHON) scripts/verify_embedding_space.py

# ------------------------------------------------------------------- docker
.PHONY: up
up: ## Start the full stack (postgres, qdrant, api, ui)
	docker compose up --build -d
	@echo "UI  http://localhost:5173"
	@echo "API http://localhost:8000/docs"

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean-volumes
clean-volumes: ## Stop the stack and delete its data volumes
	docker compose down -v

.PHONY: docker-index
docker-index: ## Index the catalogue inside the stack
	docker compose run --rm indexer

.PHONY: logs
logs: ## Tail the stack logs
	docker compose logs -f

# -------------------------------------------------------------- migrations
.PHONY: migrate
migrate: ## Apply database migrations
	cd $(BACKEND) && ../$(PYTHON) -m alembic upgrade head

.PHONY: migration
migration: ## Create a migration from model changes (M="message")
	cd $(BACKEND) && ../$(PYTHON) -m alembic revision --autogenerate -m "$(or $(M),schema change)"

.PHONY: clean
clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache \
	       $(BACKEND)/.coverage $(BACKEND)/htmlcov frontend/dist
