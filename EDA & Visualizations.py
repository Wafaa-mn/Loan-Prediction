import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Read the files
train = pd.read_csv("traindata.csv")
test = pd.read_csv("testdata.csv")


# Plotting loan status
plt.figure(1)
train['Loan_Status'].value_counts().plot.bar()
plt.suptitle("Loan Status", fontsize=14, fontweight='bold')
# Around 69% (422) out of 614 of loans were approved in the Train dataset.
plt.show()

# Figure 2: Categorical Features
plt.figure(2)

# First subplot (top-left)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Gender')
# Gender (Categorical): 80% of the applicants are males.

# Second subplot (top-right)
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
# Married (Categorical): Around 65% are married.

# Third subplot (bottom-left)
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self Employed')
# Self-employed (Categorical): Around 15% are self-employed.

# Fourth subplot (bottom-right)
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit History')
# Credit History (Categorical): Around 85% have repaid their debt.
plt.show()

# Figure 3: Applicant Income Distribution and Boxplot
plt.figure(3)

# First subplot (left side)
plt.subplot(121)
sns.histplot(train['ApplicantIncome'])

# Second subplot (right side)
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
# It can be inferred that most of the data in the distribution of applicant income is towards left which means it is not normally distributed.
# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in society.
plt.show()
plt.figure(4)
train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("Income by Education")
# The boxplot for income by education level shows a higher concentration of outliers among graduates compared to non-graduates.
plt.show()

#Coapplicant income distribution..
plt.figure(5)
plt.subplot(121)
sns.histplot(train['CoapplicantIncome'])

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16, 5))
plt.show()

# Figure 6: Loan Amount Distribution and Boxplot
plt.figure(6)

# First subplot (left side)
plt.subplot(121)
sns.histplot(train['LoanAmount'])

# Second subplot (right side)
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16, 5))
#The presence of many outliers in the loan amount distribution suggests that while most loan amounts follow a fairly normal distribution, there are some extreme values that deviate significantly from the typical loan amounts, possibly due to special cases or large loans.

plt.show()

#Bivariate analysis: We will now explore the variables with respect to the Target variable.

# Figure 7: Gender and Loan Status Relationship
plt.figure(7)
Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender = Gender.div(Gender.sum(1).astype(float), axis=0)
Gender.plot(kind="bar", stacked=True, figsize=(6, 6))

# We can infer that gender does not really have significant affect on weather the loan will be approved or not as across the two genders the proportions are the almost the same.
plt.show()

# Figure 8: Married and Loan Status Relationship
plt.figure(8)
Married = pd.crosstab(train['Married'], train['Loan_Status'])
Married = Married.div(Married.sum(1).astype(float), axis=0)
Married.plot(kind="bar", stacked=True, figsize=(6, 6))
plt.show()

# Figure 9: Number of Dependents and Loan Status Relationship
plt.figure(9)
Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
Dependents = Dependents.div(Dependents.sum(1).astype(float), axis=0)
Dependents.plot(kind="bar", stacked=True, figsize=(6, 6))
plt.show()

# Figure 10: Education and Loan Status Relationship
plt.figure(10)
Education = pd.crosstab(train['Education'], train['Loan_Status'])
Education = Education.div(Education.sum(1).astype(float), axis=0)
Education.plot(kind="bar", stacked=True, figsize=(8, 8))
plt.show()

# Figure 11: Self_Employed and Loan Status Relationship
plt.figure(11)
Self_Employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
Self_Employed = Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0)
Self_Employed.plot(kind="bar", stacked=True, figsize=(6, 6))
plt.show()

# Figure 12: Credit_History and Loan Status Relationship
plt.figure(12)
Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History = Credit_History.div(Credit_History.sum(1).astype(float), axis=0)
Credit_History.plot(kind="bar", stacked=True, figsize=(6, 6))
plt.show()

# Figure 13: Property_Area and Loan Status Relationship
plt.figure(13)
Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area = Property_Area.div(Property_Area.sum(1).astype(float), axis=0)
Property_Area.plot(kind="bar", stacked=True, figsize=(7, 7))
plt.show()



