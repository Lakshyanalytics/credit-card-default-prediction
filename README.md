# Credit Card Default Prediction
### Independent Credit Risk Modeling Project

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green) ![Methodology](https://img.shields.io/badge/Methodology-WoE%20Scorecard-orange) ![Dataset](https://img.shields.io/badge/Dataset-UCI%202016-lightgrey)

---

## Overview

This project builds a binary classification model to predict whether a credit card customer will default on their payment next month. It follows the **end-to-end credit scorecard methodology** used in the financial industry — from data cleaning through WoE-based feature engineering to logistic regression with backward elimination — and benchmarks the final model against Random Forest.

The project is structured for the **Credit Risk Framework (CRF)** context, with an emphasis on model interpretability, regulatory compliance, and explainability of individual predictions.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients (Taiwan, 2016)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

| Property | Value |
|---|---|
| Records | 30,000 (29,965 after cleaning) |
| Features | 23 raw → 16 final (post feature selection) |
| Target | Binary: 1 = default next month, 0 = no default |
| Class distribution | 78% non-default / 22% default |
| Time period | April 2005 – September 2005 |

---

## Project Pipeline

```
Data Cleaning
      │
      ▼
Outlier Treatment (Winsorization)
      │
      ▼
Information Value (IV) Analysis  →  Drop IV < 0.02
      │
      ▼
WoE Transformation
      │
      ▼
Bivariate EDA (Target Rate by Bin)
      │
      ▼
Correlation Filtering  →  Drop pairwise corr > 0.90
      │
      ▼
VIF Analysis  →  Drop VIF > 2.0 (iterative)
      │
      ▼
Train/Test Split (80/20)  →  SMOTE on train only
      │
      ▼
Logistic Regression + Backward Elimination (p < 0.05)
      │
      ▼
Odds Ratios + SHAP Explainability
      │
      ▼
Random Forest (Challenger Model)
      │
      ▼
Model Comparison + Final Selection
```

---

## Key Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 77% | 78% |
| **ROC-AUC** | **0.7663** | 0.7206 |
| **Default Recall** | **0.59** | 0.42 |
| Default Precision | 0.48 | 0.51 |
| **Default F1** | **0.53** | 0.46 |
| Macro F1 | 0.69 | 0.66 |

**Final Model: Logistic Regression** — outperforms Random Forest on every metric critical to credit risk, particularly defaulter recall (+17 percentage points) and ROC-AUC (+0.046).

---

## Why Logistic Regression Outperformed Random Forest

WoE transformation linearised all feature relationships before modeling, eliminating Random Forest's primary advantage of capturing non-linearity. The result is a simpler model that is also more accurate, more interpretable, and more suitable for a regulated credit environment.

---

## Top Predictive Features

| Rank | Feature | Odds Ratio | Direction |
|---|---|---|---|
| 1 | BILL_AMT5_woe | 2.35 | Risk ↑ |
| 2 | PAY_1_woe | 2.09 | Risk ↑ |
| 3 | EDUCATION_woe | 1.80 | Risk ↑ |
| 4 | BILL_AMT3_woe | 0.17 | Risk ↓ |
| 5 | BILL_AMT1_woe | 0.26 | Risk ↓ |

SHAP analysis confirmed **PAY_1_woe** (most recent repayment delay) as the dominant driver at individual prediction level.

---

## Methodology Highlights

- **IV/WoE Analysis** via `scorecardpy` — credit scorecard industry standard
- **Backward Elimination** — iterative p-value-based feature removal (threshold: 0.05)
- **VIF threshold of 2.0** — stricter than conventional 10, required for interpretable credit scorecard coefficients
- **SMOTE applied post-split** — avoids data leakage; test set retained at original 78/22 ratio
- **SHAP (LinearExplainer)** — individual-level prediction explainability for compliance

---

## Repository Structure

```
credit-card-default-prediction/
│
├── README.md
├── credit_card_default_prediction.ipynb   ← Main Colab notebook
└── Credit_Card_Default_Prediction_Documentation.docx  ← Full project documentation
```

---

## How to Run

1. Open `credit_card_default_prediction.ipynb` in Google Colab
2. Download the dataset from the [UCI Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and upload to Colab
3. Install dependencies:
```python
!pip install scorecardpy imbalanced-learn shap statsmodels --quiet
```
4. Run all cells in order

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scorecardpy
statsmodels
scikit-learn
imbalanced-learn
shap
```

---

## Key Thresholds Reference

| Parameter | Value | Rationale |
|---|---|---|
| IV drop threshold | 0.02 | Industry standard for useless predictors |
| Correlation threshold | 0.90 | Remove near-identical features |
| VIF threshold | 2.0 | Strict — required for scorecard interpretability |
| Backward elimination p-value | 0.05 | 95% significance threshold |
| SMOTE application | Train set only | Prevent data leakage |
| Test set ratio | 78/22 (original) | Reflects real-world conditions |

---

## Author

**Lakshya Rana**   
April 2026
