import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get data
train_df = pd.read_csv("./train.csv", dtype={"Age":np.float64},)
test_df = pd.read_csv("./test.csv", dtype={"Age":np.float64},)

# preview the data
#print(train_df.head(5))
#print("------------------------------------------------------")
#print(test_df.head(5))
#print("------------------------------------------------------")
#train_df.info()
#print("------------------------------------------------------")
#test_df.info()

# drop useless colums, drop "Cabin" for its lot of n/a
train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test_df = test_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

