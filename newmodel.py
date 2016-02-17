import pandas as pd
import csv as csv
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
test_df = test_df.drop(['Name','Ticket','Cabin'],axis=1)

# Replace "Sex" and "Embarked" with numbers
train_df["Embarked"] = train_df["Embarked"].fillna("S")
train_df["Gender"] = train_df["Sex"].map({'female':0, 'male':1}).astype(int)
test_df["Gender"] = test_df["Sex"].map({'female':0, 'male':1}).astype(int)
train_df["Embarked_num"] = train_df["Embarked"].\
		map({'S':0,'C':1,'Q':2}).astype(int)
test_df["Embarked_num"] = test_df["Embarked"].\
		map({'S':0,'C':2,'Q':2}).astype(int)

# Fill NaN of age between Mean+/-std
average_age1 = train_df["Age"].mean()
std_age1 = train_df["Age"].std()
cout_age1 = train_df["Age"].isnull().sum()

average_age2 = test_df["Age"].mean()
std_age2 = test_df["Age"].std()
cout_age2 = test_df["Age"].isnull().sum()

print(average_age1,std_age1,cout_age1)
print(average_age2,std_age2,cout_age2)


rand1 = np.random.randint(average_age1 - std_age1,\
		average_age1 + std_age1,size=cout_age1)
rand2 = np.random.randint(average_age2 - std_age2,\
		average_age2 + std_age2,size=cout_age2)

train_df["Age"][np.isnan(train_df["Age"])] = rand1
test_df["Age"][np.isnan(test_df["Age"])] = rand2

# Fill NaN of Fare in test_df, only 1 value
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].mean())


# Prepare for classifier
xtrain = train_df.drop(["Sex","Embarked","Survived"],axis=1).astype(int)
ytrain = train_df["Survived"]
testdata = test_df.drop(["PassengerId","Sex","Embarked"],axis=1).astype(int)

# Preview data
print(xtrain.head(5))
print("------------------------------------------------------")
print(testdata.head(5))
print("------------------------------------------------------")
xtrain.info()
print("------------------------------------------------------")
testdata.info()

# Random Forests
forest = RandomForestClassifier(n_estimators=100)
forest.fit(xtrain,ytrain)
output = forest.predict(testdata).astype(int)
print(forest.score(xtrain,ytrain))
print(output)

# Output to a file
pred_f = open('newmodel.csv','w+')
pred_f_obj = csv.writer(pred_f)
pred_f_obj.writerow(["PassengerId", "Survived"])

for i in range(0,len(output)):
	pred_f_obj.writerow([test_df.PassengerId[i],output[i]])
pred_f.close

# SVM
svc = SVC()
svc.fit(xtrain,ytrain)
output_svc = svc.predict(testdata).astype(int)
print(svc.score(xtrain,ytrain))

# Output to a file
predsvc_f = open('newmodel_svc.csv','w+')
predsvc_f_obj = csv.writer(predsvc_f)
predsvc_f_obj.writerow(["PassengerId", "Survived"])

for i in range(0,len(output)):
	predsvc_f_obj.writerow([test_df.PassengerId[i],output[i]])
predsvc_f.close


