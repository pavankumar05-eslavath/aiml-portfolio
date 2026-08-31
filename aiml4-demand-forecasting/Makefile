PY ?= python

.PHONY: help all install download panel sample audit eda baselines train classical \
        inventory scenarios forecast report notebooks app test lint clean

help:
	@echo "make all         audit -> eda -> baselines -> train -> classical -> inventory -> scenarios -> forecast -> report"
	@echo "make download    fetch the M5 source files (~133 MB, public mirror)"
	@echo "make panel       build the modelling panel from the source files"
	@echo "make sample      generate the synthetic stand-in panel (no network)"
	@echo "make data        panel if possible, else synthetic stand-in"
	@echo "make audit       time-series data quality + leakage audit"
	@echo "make eda         exploratory figures"
	@echo "make baselines   naive / seasonal naive / MA / Croston / SBA benchmarks"
	@echo "make train       walk-forward validation, held-out test, persist model"
	@echo "make classical   ETS and SARIMA on a stratified sample"
	@echo "make inventory   safety stock, reorder points, policy simulation"
	@echo "make scenarios   service-level scenarios + cost sensitivity"
	@echo "make forecast    example forecast + reorder recommendation"
	@echo "make report      write reports/model_report.md"
	@echo "make app         run the Streamlit decision dashboard"
	@echo "make test        test suite"
	@echo "make lint        ruff"
	@echo "make clean       remove generated artefacts"

all: data
	$(PY) -m src.run all

install:
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) -m src.data.download

panel:
	$(PY) -c "from src.config import load_config; from src.data.loader import build_panel; \
p=build_panel(load_config(), force=True); print(f'panel: {len(p):,} rows, {p.series_id.nunique()} series')"

sample:
	$(PY) -m src.data.generate_sample

# Prefer the real data; fall back to the synthetic panel so the pipeline always has
# something to run on (data is gitignored).
data:
	@test -f data/processed/panel.parquet \
		|| ($(PY) -m src.data.download && $(MAKE) panel) \
		|| $(PY) -m src.data.generate_sample

audit:      ; $(PY) -m src.run audit
eda:        ; $(PY) -m src.run eda
baselines:  ; $(PY) -m src.run baselines
train:      ; $(PY) -m src.run train
classical:  ; $(PY) -m src.run classical
inventory:  ; $(PY) -m src.run inventory
scenarios:  ; $(PY) -m src.run scenarios
forecast:   ; $(PY) -m src.run forecast
report:     ; $(PY) -m src.run report

notebooks:
	$(PY) tools/build_notebooks.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 notebooks/*.ipynb

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests tools app

clean:
	rm -rf reports/run_results.json reports/model_report.md reports/data_audit.md \
		reports/figures/*.png models/*.joblib models/*.json \
		data/processed/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
