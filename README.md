# Bankruptcy Risk Scoring

> How do you rank bankruptcy risk when fewer than one in twenty training firms failed?

**Competition study · stored cross-validation output**

An XGBoost classifier ranks firms using 64 financial ratios and evaluates discrimination with stratified cross-validation.

[Case study](https://daniel-kang-vs.github.io/projects/bankruptcy.html) · [Source evidence](https://github.com/daniel-kang-vs/bankruptcy-prediction/blob/main/571_final.ipynb) · [Live dashboard](https://daniel-kang-vs.github.io/bankruptcy-prediction/)

[![Dashboard preview](https://raw.githubusercontent.com/daniel-kang-vs/daniel-kang-vs.github.io/main/pngs/project-bankruptcy.png)](https://daniel-kang-vs.github.io/bankruptcy-prediction/)

## Highlights

- **10,000 training firms**
- **4.73% positive class**
- **0.907 mean CV AUC**

## The problem

A model can appear accurate by calling almost every company solvent. Risk screening needs a ranking metric and a clear account of class imbalance.

## Data & scope

The committed training CSV contains 10,000 rows, 64 financial-ratio columns, and 473 positive labels. The test file contains 8,000 unlabeled rows.

## How it works

1. Inspect the class balance and missing financial ratios.
2. Train the raw-input XGBoost model with regularization, row sampling and feature sampling.
3. Evaluate with five shuffled stratified folds using random seed 42.
4. Fit the final raw-input model and export test probabilities for the competition.

## Key technical decision

Use ROC-AUC to evaluate ranking independently of one arbitrary operating threshold. Although the notebook explores winsorization and KNN imputation, the stored 0.907 score comes from X_full_raw, not that preprocessing path.

## Results & interpretation

The stored fold AUCs are 0.918, 0.936, 0.896, 0.912 and 0.875, averaging 0.9073. The historical competition placement is reported as third of 53 teams; a leaderboard artifact is not bundled in the repository.

## Limitations & next steps

Cross-validation is not prospective validation. A deployment would need calibration, drift checks, and a threshold based on the cost of missed failures versus false alarms. No preprocessing lift or operational impact is established by the saved output.

## Evidence & reproducibility

571_final.ipynb stores the five-fold scores for the raw-input model. Dashboard class counts and missingness are derived from train.csv.

## Repository guide

- [`571_final.ipynb`](571_final.ipynb) — analysis implementation.
- [`docs/`](docs/) — static dashboard, styles, and browser-side analysis.

## Run or inspect

Clone this repository and run from its root:

```bash
pip install jupyter pandas numpy scipy scikit-learn xgboost
jupyter notebook 571_final.ipynb
```

Notebook commands require the referenced CSV files and compatible dependencies; recorded results have not been retrained for this documentation refresh.

## Technology

Python · XGBoost · scikit-learn · pandas

## Refresh the descriptive dashboard

Run `python scripts/refresh_dashboard_data.py` from the repository root, then serve `docs/` with `python3 -m http.server 8000 --directory docs`. Source hashes and interpretation notes are in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Dashboard exploration

The KPI strip reports stored fold scores, the positive-class share, and missing numeric cells divided by 10,000 × 64. The data-quality selector changes the displayed field detail, not the overall population. CSV export contains exactly the displayed chart series.
