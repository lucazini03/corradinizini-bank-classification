# corradinizini-bank-classification

## Project Overview
This project solves a binary classification task on a bank marketing dataset.
The objective is to predict whether a client will subscribe to a term deposit.

The project was developed for the Machine Learning course Kaggle challenge.

---

## Objectives
- Explore the dataset
- Perform feature engineering
- Train multiple ML models
- Compare performance using ROC-AUC
- Build an ensemble solution

---

## Dataset Description
The dataset contains client and campaign information:

Client data:
- age
- job
- marital status
- education
- balance
- housing loan
- personal loan

Campaign data:
- duration
- pdays
- campaign
- previous contacts

Target variable:
- subscription to term deposit

Train size: 750k rows  
Test size: 250k rows  

---

## Feature Engineering
We created new features including:
- Binary indicator for previous contact (pdays == -1)
- Transformations of skewed numerical features
- Interaction features
- Encoding categorical variables

---

## Models Used
- Random Forest baseline (+ to see feature importance)
- XGBoost
- CatBoost
- LightGBM
- Stacking with Logistic Regression

All models used stratified 10-fold cross-validation.

---

## Results
Final ROC-AUC score: **0.97478**  
Leaderboard placement: **Top 14%**

---

## Repository Structure
corradinizini-bank-classification/
│
├── data-software/
│ ├── README.md
│ ├── requirements.txt
│ ├── bank_dataset_ml.ipynb
│ ├── sample_submission.csv
│ ├── submission.csv
│ ├── test.csv
│ └── train.csv
├── doc/
│ ├── imgs/
│ ├── sources/
│ ├── presentation.tex
│ └── presentation.pdf
└── README.md

---

## References
- Kaggle Bank Marketing Competition
- Scikit-Learn documentation
- XGBoost, LightGBM, CatBoost documentation
- Course materials and lectures