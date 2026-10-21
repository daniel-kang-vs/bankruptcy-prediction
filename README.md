# Bankruptcy Risk Scoring — MGMT 571

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![XGBoost](https://img.shields.io/badge/XGBoost-binary:logistic-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN%20%2B%20CV-f7931e?logo=scikit-learn)

> **Competition result:** binary bankruptcy classification from 64 financial ratios. Mean 5-fold ROC-AUC **0.907**. **3rd of 53 teams** in Purdue MGMT 571.

---

## Project Highlights

- Accuracy is the wrong score. Failing firms are the minority, so a model that always says “solvent” looks strong and ranks nobody
- Outliers are capped with **winsorization at the 0.5th and 99.5th percentiles** before imputation, so a few broken ratios do not set the scale
- Missing ratios are filled with **5-nearest neighbors**, which keeps relationships between ratios instead of dropping them to a column mean
- One regularized XGBoost model, judged only by stratified ROC-AUC

---

## Problem Statement

Predict `class = 1` (bankrupt) or `class = 0` (solvent) from financial-statement ratios.

| File | Rows | Columns | Role |
|------|------|---------|------|
| `train.csv` | 10,000 | 65 | 64 ratios + `class` |
| `test.csv` | 8,000 | 65 | Features for submission |
| `bankruptcy_sample_submission.csv` | — | — | Submission format |

The ratios cover liquidity, leverage, profitability, and efficiency. They contain missing values and extreme tails. The leaderboard metric is ROC-AUC, which rewards ranking distressed firms above healthy ones.

---

## Preprocessing

```
1. Winsorize every ratio at the 0.5th and 99.5th percentiles
2. Impute remaining missing values with KNN (k = 5) on the winsorized matrix
3. Fit XGBoost on the completed training matrix
4. Score with stratified 5-fold ROC-AUC
```

Winsorization happens first on purpose. If a ratio has a handful of absurd values, those points dominate a distance-based imputer and the tree splits that follow.

---

## Model

XGBoost `binary:logistic`:

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 1,000 |
| `learning_rate` | 0.02 |
| `max_depth` | 6 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` (L1) | 0.1 |
| `reg_lambda` (L2) | 1.0 |

L1 and L2 are both on because the 64 ratios are correlated. Depth 6 with a low learning rate is the bias-variance choice: deep enough to use interactions between ratios, slow enough not to memorize the minority class.

---

## Results

| Fold | ROC-AUC |
|------|---------|
| 1 | 0.9179 |
| 2 | 0.9359 |
| 3 | 0.8963 |
| 4 | 0.9116 |
| 5 | 0.8746 |
| **Mean** | **0.9073** |

Placement: **3rd of 53 teams**.

Fold 5 (0.875) is the reminder that a single split would have overstated the model. The mean is what was trusted.

---

## Key Technical Decisions

1. **ROC-AUC, not accuracy.** The decision a credit team needs is a ranking. Accuracy rewards predicting the majority class.

2. **Winsorize, then impute.** Clipping after KNN lets the extreme points distort the neighbors. Clipping first keeps the neighborhood about typical firms.

3. **KNN instead of a column median.** A missing leverage ratio is more plausibly “like similar firms” than “the median of everyone.” Median fill throws away that structure.

4. **Regularize the trees.** With correlated ratios, an unpenalized booster will split on noise that happens to separate a few bankruptcies. L1 and L2 are there to stop that.

5. **What this does not set.** AUC does not choose the cutoff a lender should use. That cutoff depends on the cost of a missed bankruptcy versus a false alarm, which was not the competition objective.

---

## Repository Structure

```
├── 571_final.ipynb
├── train.csv
├── test.csv
├── bankruptcy_sample_submission.csv
├── Output/
│   └── Best_AUC.csv
├── docs/                 # Live writeup
└── README.md
```

---

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook 571_final.ipynb
```

Training data is in the repository. Cross-validation is the number to read; the test labels were held out by the competition.

**[Live demo](https://daniel-kang-vs.github.io/bankruptcy-prediction/)**

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| XGBoost | Gradient-boosted classifier |
| scikit-learn | KNN imputation, stratified K-fold, ROC-AUC |
| pandas / NumPy | Ratio tables and winsorization |
| matplotlib / seaborn | Fold and distribution charts |
