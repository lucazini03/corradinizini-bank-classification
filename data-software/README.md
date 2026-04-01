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
1. **Global Constants, Reproducibility, Imports** — Seeds, shared constants (`RANDOM_SEED`, `N_SPLITS`, `TEST_SIZE`), and all library imports.
2. **Load Data** — Load `train.csv`, `test.csv`, and `sample_submission.csv` from the chosen environment; drop the non-informative `id` column.
3. **Missing Value Check** — Confirm no missing values before making any modelling decisions.
4. **Exploratory Data Analysis (EDA)** — Statistical summary, numerical correlation matrix, target class distribution, and target rate broken down by each categorical feature.
5. **Feature Engineering** — Three feature sets are defined (raw, v1, v2) and evaluated on a 30 % subsample via quick 3-fold CV. The winning set is selected automatically and used for all downstream steps. New features include call-intensity ratios, binary hang-up and fatigue flags, and a job × education interaction.
6. **Preprocessing Pipelines** — Skewness visualisation, feature column assignment to transformation buckets (log1p, cube-root, bespoke pdays handler, scale-only), custom picklable transformers, and a `build_preprocessor()` factory that supports both one-hot and target encoding strategies.
7. **Prepare X / y** — Assemble the final feature matrices `X`, `y`, and `X_test_final` using the selected feature set.
8. **CV Helper** — `run_cv()` utility: stratified k-fold cross-validation returning OOF predictions, per-fold AUC scores, and averaged test-set predictions.
9. **Baseline: Random Forest** — Untuned RF on a hold-out split (sanity check), then full stratified CV to produce the anchor scores for the ablation study.
10. **XGBoost: Hyperparameter Tuning + CV** — GridSearchCV on a 30 % subsample to find best `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`; full CV with the best parameters and 1 000 estimators.
11. **LightGBM: Hyperparameter Tuning + CV** — Same pattern as step 10, tuning `num_leaves`, `learning_rate`, and `subsample`.
12. **Random Forest: Hyperparameter Tuning + CV** — GridSearchCV over `n_estimators`, `max_depth`, and `min_samples_leaf`; full CV for use as a base learner in stacking.
13. **Stacking Ensemble** — Two Logistic Regression meta-learners: one trained on OOF predictions from all three base models (XGB + LGBM + RF), and one trained on XGB + LGBM only (no RF) — both saved for comparison.
14. **Final Model Comparison Table** — Side-by-side CV AUC (mean ± std) for all six variants: RF baseline, RF tuned, XGBoost, LightGBM, stacking with RF, stacking without RF.
15. **Ablation Study** — Step-by-step table isolating the contribution of each decision: raw features → feature engineering → naive ensemble → learned stacking with RF (step D) → learned stacking without RF (step E, delta measured against the naive ensemble to test whether removing the RF recovers the gap).
16. **Submission File** — Automatically selects the best model by CV AUC across all variants and writes `submission.csv`.