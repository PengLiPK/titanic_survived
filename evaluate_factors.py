import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('whitegrid')
matplotlib.use('TKAgg')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get data
train_df = pd.read_csv("./train.csv", dtype={"Age":np.float64},)

# preview the data
#print(train_df.head(5))
#print("------------------------------------------------------")
#train_df.info()

# Embarked
train_df["Embarked"] = train_df["Embarked"].fillna("S")
sns.factorplot(x="Embarked",y="Survived",data=train_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x="Embarked", data=train_df,ax=axis1)
sns.countplot(x="Survived", hue="Embarked", data=train_df, order=[1,0], ax=axis2)

embark_mean = train_df[["Embarked", "Survived"]]\
		.groupby(["Embarked"], as_index=False).mean()
sns.barplot(x="Embarked", y="Survived", data=embark_mean, \
		order=["S","C","Q"],ax=axis3)

plt.show()

# Fare
