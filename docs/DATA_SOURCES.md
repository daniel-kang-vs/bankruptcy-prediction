# Dashboard data provenance

- Source: [`train.csv`](../train.csv).
- SHA-256 of reviewed input: `291aa5817bdb2b4dc897f5390c9b7dc1f31ca4a755561b914408b55b6f4e5183`.
- Refresh descriptive counts from the repository root: `python scripts/refresh_dashboard_data.py`.
- `data.json` stores the displayed chart series. Expand each chart's table to inspect exact values.
- The refresh script does not train or validate a predictive model. Captions and result claims need review if the input changes.

Five fold ROC-AUC values are transcribed from the saved output of `571_final.ipynb`, using the raw-input XGBoost model. They are historical cross-validation scores, not a new run.
