# Software and Data

## Kaggle Challenge
Bank Marketing Binary Classification challenge.

The goal is to predict whether a client subscribes to a term deposit.

---

## Installation

Requirements:
- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- catboost

Install with the following commands:

- `poetry lock`
- `poetry install`

---

## Running the Project

The notebook execution is way faster if you use a GPU, either in Colab/Kaggle or in local with a compatible GPU.

1. Open the notebook bank_dataset_ml.ipynb
2. Choose whether you want to run in local or Colab or Kaggle environment, and adjust the first cell accordingly.
3. Run all cells in order.
4. The final submission file will be generated automatically.

---

## Notebook Workflow
1. Data loading and inspection
2. Missing value check
3. Data exploration
4. Feature engineering
5. Numerical features
    6. Handle skewed distributions
7. Categorical features
    8. Encoding categorical variables
9. ML models
    9. Random Forest as a baseline model and feature importance
    10. XGboost
    11. CatBoost
    12. LightGBM
    13. Ensemble stacking
14. Submission file generation