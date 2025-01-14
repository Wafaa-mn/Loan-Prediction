import pandas as pd
import numpy as np    #For mathmatical manipulation
import seaborn as sns #for data visulization
import matplotlib.pyplot as plt  #for plotting graphs
plt.show()
import warnings
warnings.filterwarnings("ignore")



#read the files
train = pd.read_csv("traindata.csv")
test = pd.read_csv("testdata.csv")
print(train.shape)
# Make copies of the datasets
train_original = train.copy()
test_original = test.copy()


train.isnull().sum()
## Treating the missing values using the median for numerical values since we have found outliers,
# and mode for categorical variables.

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Married'].fillna(train['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# Save the cleaned data to a CSV file
train.to_csv('cleaned_train_data.csv', index=False)
test.to_csv('cleaned_test_data.csv', index=False)



