PY ?= python

.PHONY: help all install data download profile naive audited coverage export test lint clean

help:
	@echo "make all        generate data and run every stage"
	@echo "make data       generate the defect-preserving stand-in (no download)"
	@echo "make download   fetch the real 3,044-row Kaggle CSV (CC0, GitHub mirror)"
	@echo "make profile    profile the file and locate its defects"
	@echo "make naive      the baseline: the obvious, uncleaned analysis"
	@echo "make audited    the corrected analysis, next to the baseline"
	@echo "make coverage   validate against published industry totals"
	@echo "make export     write data/funding_clean.csv and data/investor_deals.csv"
	@echo "make test       test suite"
	@echo "make lint       ruff"
	@echo "make clean      remove generated data and outputs"

all: data
	$(PY) -m src.run --save-figures all

install:
	$(PY) -m pip install -r requirements.txt

data:
	@test -f data/startup_funding.csv \
		|| test -f data/startup_funding_sample.csv \
		|| $(PY) -m data.generate_sample

download:
	$(PY) -m data.download

profile:
	$(PY) -m src.run profile

naive:
	$(PY) -m src.run naive

audited:
	$(PY) -m src.run --save-figures audited

coverage:
	$(PY) -m src.run coverage

export:
	$(PY) -m src.run export

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src data tests

clean:
	rm -rf data/startup_funding_sample.csv data/funding_clean.csv \
		data/investor_deals.csv outputs .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
