PY ?= python

.PHONY: help all install data download profile article diagnose improved audit test lint clean

help:
	@echo "make all        generate data and run every stage"
	@echo "make data       generate the small PaySim stand-in (no download)"
	@echo "make download   fetch the real 480 MB PaySim CSV from Drive"
	@echo "make profile    data profile; checks the article's claims"
	@echo "make article    the published pipeline, reproduced as written"
	@echo "make diagnose   why its XGBoost result no longer reproduces"
	@echo "make improved   temporal split, class weighting, PR-AUC, tuned threshold"
	@echo "make audit      the three-clause rule that beats the models"
	@echo "make test       test suite"
	@echo "make lint       ruff"
	@echo "make clean      remove generated data and outputs"

all: data
	$(PY) -m src.run --save-figures all

install:
	$(PY) -m pip install -r requirements.txt

data:
	@test -f data/onlinefraud.csv \
		|| test -f data/paysim_synthetic.csv \
		|| $(PY) -m data.generate_paysim

download:
	$(PY) -m data.download

profile:
	$(PY) -m src.run profile

article:
	$(PY) -m src.run article

diagnose:
	$(PY) -m src.run diagnose

improved:
	$(PY) -m src.run --save-figures improved

audit:
	$(PY) -m src.run audit

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src data tests

clean:
	rm -rf data/paysim_synthetic.csv outputs .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
