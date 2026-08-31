PY ?= python

.PHONY: help all install download sample data audit eda train explain predict report \
        notebooks test lint clean

help:
	@echo "make all         run every stage: audit -> eda -> train -> explain -> predict -> report"
	@echo "make download    fetch the real Telco churn dataset (7,043 rows)"
	@echo "make sample      generate the synthetic stand-in (no network)"
	@echo "make audit       data quality and leakage audit -> reports/data_audit.md"
	@echo "make eda         exploratory figures -> reports/figures/"
	@echo "make train       train, tune, calibrate, evaluate, persist the model"
	@echo "make explain     SHAP global + per-customer explanations"
	@echo "make predict     score the example customer"
	@echo "make report      write reports/model_report.md"
	@echo "make notebooks   execute all six notebooks, failing on any error"
	@echo "make test        test suite"
	@echo "make lint        ruff"
	@echo "make clean       remove generated artefacts"

all: data
	$(PY) -m src.run all

install:
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) -m src.data.download

sample:
	$(PY) -m src.data.generate_sample

# Prefer the real dataset; fall back to the synthetic stand-in so the pipeline
# always has something to run on (the CSVs are gitignored).
data:
	@test -f data/raw/telco_churn.csv \
		|| test -f data/raw/telco_churn_sample.csv \
		|| $(PY) -m src.data.download \
		|| $(PY) -m src.data.generate_sample

audit:
	$(PY) -m src.run audit

eda:
	$(PY) -m src.run eda

train:
	$(PY) -m src.run train

explain:
	$(PY) -m src.run explain

predict:
	$(PY) -m src.run predict

report:
	$(PY) -m src.run report

notebooks:
	$(PY) tools/build_notebooks.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/*.ipynb

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests tools

clean:
	rm -rf reports/run_results.json reports/model_report.md reports/data_audit.md \
		reports/figures/*.png models/*.joblib models/*.json \
		data/raw/telco_churn_sample.csv data/processed/* \
		.pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
