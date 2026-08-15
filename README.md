# Machine Learning Assignment 2 – Classification Model Explorer

## a. Problem statement

Implement and compare the required classification models on one common public classification dataset, then demonstrate the trained models through an interactive Streamlit application.

## b. Dataset description

The Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository (Dataset ID 17) contains 569 instances and 30 real-valued features. The target variable is binary, with malignant and benign classes. An 80/20 stratified train/test split with random_state=42 is used. Malignant is treated as the positive class for AUC, Precision, Recall and F1.

Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## c. GitHub Repository Link

https://github.com/2025ad05049-Sravani/ml-assignment-2-classification-dashboard

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9649 | 0.9944 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline; scaling makes optimization stable across features. |
| Decision Tree | Captures non-linear interactions and is interpretable; depth restriction helps control overfitting. |
| kNN | Distance-based model that benefits from scaling; performance depends on neighbourhood size. |
| Naive Bayes | Fast and simple, but Gaussian/conditional-independence assumptions can be restrictive for correlated features. |
| Random Forest (Ensemble) | Combines many trees to reduce variance and model non-linear relationships effectively. |
| **Overall Best Model** | **Logistic Regression**, based on the highest F1 score on the held-out test set. |

## Streamlit App

Live Streamlit App: https://ml-breast-cancer-classifier.streamlit.app/

## Repository structure

```text
ml-assignment-2-classification-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── breast_cancer_wisconsin.csv
└── model/
    ├── README.md
    ├── decision_tree.joblib
    ├── decision_tree.py
    ├── knn.joblib
    ├── knn.py
    ├── logistic_regression.joblib
    ├── logistic_regression.py
    ├── metadata.json
    ├── naive_bayes.joblib
    ├── naive_bayes.py
    ├── random_forest.py
    ├── random_forest_ensemble.joblib
    └── train_models.py
```

The `.py` files are the inspectable model source implementations. The `.joblib` files are serialized trained estimators used by the Streamlit app.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
