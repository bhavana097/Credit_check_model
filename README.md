# Credit_check_model
This project aims to predict credit card defaults using various machine learning models. The dataset includes demographic and financial information, and the target variable indicates whether a customer will default on their payment in the next month.
Data Preprocessing

Original Features

The dataset originally contained the following columns:

Demographic Variables: ID, SEX, EDUCATION, MARRIAGE, AGE

Payment History: PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6

Billing Amounts: BILL_AMT1 - BILL_AMT6

Payment Amounts: PAY_AMT1 - PAY_AMT6

Target Variable: default.payment.next.month

Feature Engineering

To reduce multicollinearity and enhance model interpretability, we dropped redundant features and created new ones:

Dropped Features:

BILL_AMT1 - BILL_AMT6

PAY_AMT1 - PAY_AMT6

New Features:

avg_delay, max_delays, total_delays

Utilization_1 - Utilization_6, avg_utilization

payment_ratio2 - payment_ratio6

payment_trend1 - payment_trend5

Bill_trend1 - Bill_trend5

Combining Features with High VIF
We identified multicollinearity using Variance Inflation Factor (VIF) and combined related features:

x_train_resampled["new_combined_feature"] = x_train_resampled["SEX"] + x_train_resampled["EDUCATION"]
x_train_resampled["new_combined_feature2"] = x_train_resampled["MARRIAGE"] + x_train_resampled["AGE"]
X_combined = x_train_resampled.drop(columns=["SEX", "EDUCATION", "MARRIAGE", "AGE"])

Handling Class Imbalance

Due to an imbalance in the target classes, we applied Synthetic Minority Over-sampling Technique (SMOTE) to create a balanced dataset before training.

Model Training & Evaluation

1. Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

Results:

Accuracy: 64%
Precision (Class 1): 0.33
Recall (Class 1): 0.63
F1-score (Class 1): 0.43

2. Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

Results:

Accuracy: 72%
Precision (Class 1): 0.41
Recall (Class 1): 0.59
F1-score (Class 1): 0.48

3. XGBoost Classifier

from xgboost import XGBClassifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=79)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

Results:

Accuracy: 72%
Precision (Class 1): 0.41
Recall (Class 1): 0.57
F1-score (Class 1): 0.48

Observations

Logistic Regression underperformed due to its simplicity in capturing non-linear relationships.

Random Forest & XGBoost improved accuracy but still struggled with class imbalance, particularly in Recall for Class 1 (default cases).

Feature Engineering & SMOTE helped mitigate some issues but did not significantly boost performance beyond 72% accuracy.

Logloss was used as an evaluation metric for XGBoost due to its effectiveness in handling probabilistic predictions.

Next Steps

Hyperparameter tuning for XGBoost and Random Forest.

Experiment with different resampling techniques (e.g., ADASYN, Tomek Links).

Try alternative algorithms (e.g., LightGBM, CatBoost).

Use cost-sensitive learning to emphasize minority class predictions.
