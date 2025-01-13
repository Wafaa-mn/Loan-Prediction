import pandas as pd
import numpy as np    #For mathematical manipulation
import matplotlib.pyplot as plt  #for plotting graphs
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

# Load the data
cleaned_train = pd.read_csv('cleaned_train_data.csv')
cleaned_test = pd.read_csv('cleaned_test_data.csv')
original_test=pd.read_csv('testdata.csv')

# Recalculate LoanAmount_log to handle skewness (we did this because we are going to use logistic regression)
cleaned_train['LoanAmount_log'] = np.log(cleaned_train['LoanAmount'])
cleaned_test['LoanAmount_log'] = np.log(cleaned_test['LoanAmount'])

# Drop 'Loan_ID' as it doesn't affect our prediction
cleaned_train = cleaned_train.drop(['Loan_ID'], axis=1)
cleaned_test = cleaned_test.drop(['Loan_ID'], axis=1)

# Sklearn requires the target variable to be in a different dataset
x = cleaned_train.drop(['Loan_Status'], axis=1)
y = cleaned_train['Loan_Status']
# Check the columns in the cleaned_train dataset
#print(cleaned_train.columns)

# Encode target variable
label_encoder = LabelEncoder()
y = pd.Series(label_encoder.fit_transform(y))


# Create dummy variables for categorical columns
x = pd.get_dummies(x)
cleaned_train = pd.get_dummies(cleaned_train)
cleaned_test = pd.get_dummies(cleaned_test)






# Initialize StratifiedKFold with 5 splits
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

i = 1  # To track fold number
for train_index, test_index in kf.split(x, y):
    print(f'Fold {i} of {kf.n_splits}')

    # Split the data into training and validation sets
    xtr, xvl = x.iloc[train_index], x.iloc[test_index]  # Training and validation input
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]  # Training and validation target

    # Train and evaluate Logistic Regression
    log_model = LogisticRegression(random_state=1)
    log_model.fit(xtr, ytr)
    log_pred = log_model.predict(xvl)
    log_score = accuracy_score(yvl, log_pred)
    print(f"Logistic Regression Accuracy for Fold {i}: {log_score}")

    # Train and evaluate Decision Tree
    dt_model = DecisionTreeClassifier(random_state=1)
    dt_model.fit(xtr, ytr)
    dt_pred = dt_model.predict(xvl)
    dt_score = accuracy_score(yvl, dt_pred)
    print(f"Decision Tree Accuracy for Fold {i}: {dt_score}")

    # Train and evaluate Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
    rf_model.fit(xtr, ytr)
    rf_pred = rf_model.predict(xvl)
    rf_score = accuracy_score(yvl, rf_pred)
    print(f"Random Forest Accuracy for Fold {i}: {rf_score}")

    # Train and evaluate XGBoost
    xgb_model = XGBClassifier(random_state=1, use_label_encoder=False)
    xgb_model.fit(xtr, ytr)
    xgb_pred = xgb_model.predict(xvl)
    xgb_score = accuracy_score(yvl, xgb_pred)
    print(f"XGBoost Accuracy for Fold {i}: {xgb_score}")

    i += 1

# Make predictions for the test dataset using the final XGBoost model
xgb_model.fit(x, y)
final_pred = xgb_model.predict(cleaned_test)

submission = pd.read_csv('sample_submission.csv')
print(submission.shape)

submission['Loan_Status'] = final_pred
submission['Loan_ID'] = original_test['Loan_ID']
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('final_predictions.csv', index=False)
print("Final predictions saved to 'final_predictions.csv'")


