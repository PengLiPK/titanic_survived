import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv as csv

# Training data cleanup
# read data, header row is 0.
df = pd.read_csv('./train.csv', header = 0)

df['Gender'] = df['Sex'].map({'female':0, 'male': 1}).astype(int)

# median ages of gender and class
median_ages = np.zeros((2,3))

for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = df[(df.Gender == i) & (df.Pclass == j+1)] \
				['Age'].dropna().median()

# Fill age with median_ages
df['AgeFill'] = df['Age']

for i in range(0,2):
	for j in range(0,3):
		df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
				'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

# Drop useless column
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

df = df.drop(['Age'],axis=1)

# Drop rows with missing values
df = df.dropna()

# Covert df to a Numpy array for Machine learning
train_data = df.values


# Testing data cleanup
# read data, header row is 0.
dft = pd.read_csv('./test.csv', header = 0)

dft['Gender'] = dft['Sex'].map({'female':0, 'male': 1}).astype(int)

# median ages of gender and class
median_ages = np.zeros((2,3))

for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = dft[(dft.Gender == i) & (dft.Pclass == j+1)] \
				['Age'].dropna().median()


# Fill age with median_ages
dft['AgeFill'] = dft['Age']

for i in range(0,2):
	for j in range(0,3):
		dft.loc[(dft.Age.isnull()) & (dft.Gender == i) & (dft.Pclass == j+1),\
				'AgeFill'] = median_ages[i,j]

dft['AgeIsNull'] = pd.isnull(dft.Age).astype(int)
dft['FamilySize'] = dft['SibSp'] + dft['Parch']
dft['Age*Class'] = dft.AgeFill * dft.Pclass

# Drop useless column
dft = dft.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
dft = dft.drop(['Age'],axis=1)
# Drop rows with missing values
print(dft[df.isnull()])
#dft = dft.dropna()

# Covert dft to a Numpy array for Machine learning
test_data = dft.values

print("Training.....")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::, 2::], train_data[0::,1])

print("Predictin....")
output = forest.predict(test_data[0::, 1::]).astype(int)

print(output)

# Output to a file
pred_f = open('forestmodel.csv','w+')
pred_f_obj = csv.writer(pred_f)
pred_f_obj.writerow(["PassengerId", "Survived"])

for i in range(0,len(output)):
	pred_f_obj.writerow([int(test_data[i,0]),output[i]])

pred_f.close
