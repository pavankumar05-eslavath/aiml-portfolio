# Notebooks

These notebooks **demonstrate the analysis**. They do not contain the
implementation — every function they call lives in `src/`, so the notebook and the
production pipeline cannot drift apart.

Run them after fetching the data:

```bash
make download          # or: make sample
jupyter lab notebooks
```

Or execute them headlessly:

```bash
make notebooks         # runs all six with nbconvert and fails on any error
```

| notebook | what it shows |
|---|---|
| `01_data_audit.ipynb` | Schema, data dictionary, missing/duplicate/invalid analysis, and the executable leakage probe. |
| `02_eda.ipynb` | The six business questions and the figures that answer them. |
| `03_feature_engineering.ipynb` | Each engineered feature, its rationale, and the check that it is row-wise and stateless. |
| `04_modeling.ipynb` | Baselines first, then tuned models, with the paired test asking whether complexity is earned. |
| `05_model_evaluation.ipynb` | Metric suite, calibration, split stability, threshold optimisation and sensitivity. |
| `06_explainability.ipynb` | Global SHAP, per-customer explanations in plain language, risk segments and the decision layer. |

`04` and `05` retrain, which takes a couple of minutes. `06` loads the persisted
bundle, so run `make train` first if you have not.
