# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave a service (churn) using customer account information and service usage features.

## Project Overview

Customer churn is an important business problem. Identifying customers who may leave can help businesses understand retention challenges and plan customer support strategies.

This project uses a Random Forest classifier to predict customer churn and evaluates its performance using several classification metrics.

## Dataset

- **Source:** [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Original records:** 7,043 customers
- **Records used:** 7,032 after handling invalid `TotalCharges` values
- **Target variable:** `Churn`
- **Excluded features:** `customerID`

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Convert `TotalCharges` to a numeric data type.
3. Remove records with invalid total charges.
4. Separate features from the target variable.
5. Encode categorical features using `OneHotEncoder`.
6. Split the data into training and testing sets.
7. Train a `RandomForestClassifier`.
8. Evaluate predictions and visualize results.

## Model Evaluation

Results from the current experiment:

| Metric | Score |
|---|---:|
| Accuracy | 0.761 |
| Precision | 0.543 |
| Recall | 0.639 |
| F1-score | 0.587 |
| ROC-AUC | 0.814 |

The model identifies a portion of customers who churn, but it also produces false positives and misses some customers who actually churn. These results leave room for further improvement.

## Generated Outputs

- `confusion_matrix.png` — visualizes correct and incorrect predictions.
- `feature_importance.png` — displays the top 10 features ranked by model importance.
- `feature_importance.csv` — exports the feature importance values.

Feature importance indicates how the model uses features for prediction; it does not establish causation.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Download the CSV dataset and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root directory.

### 3. Run the project

```bash
python main.py
```

## Future Improvements

- Compare Random Forest with Logistic Regression and Gradient Boosting.
- Tune the classification threshold to balance precision and recall.
- Use cross-validation and hyperparameter tuning.
- Investigate class imbalance and feature importance.
- Evaluate business-focused retention strategies.

## Disclaimer

This project is for educational purposes. Predictions are estimates and should not be treated as guaranteed customer behavior.
