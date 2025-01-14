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




##train['Loan_Status'].value_counts(normalize=True).plot.bar()

# Create the figure
plt.figure(1)

# First subplot (top-left)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Gender')

# Second subplot (top-right)
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')

# Third subplot (bottom-left)
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self Employed')

# Fourth subplot (bottom-right)
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit History')

plt.figure(2)

# First subplot (left side)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])

# Second subplot (right side)
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))

plt.show()  # This will display the plot

Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender = Gender.div(Gender.sum(1).astype(float), axis=0)  # Ensure the operation's result is saved
Gender.plot(kind="bar", stacked=True, figsize=(4, 4))  # Use the `.plot()` method of the DataFrame
plt.show()  # Displays the plot


print(train.isnull().sum())

