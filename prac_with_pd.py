import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as P

# read data, header row is 0.
df = pd.read_csv('./train.csv', header = 0)

#print(df.head(3))
#print(type(df))
#print(df.dtypes)
#print(df.info())
#print(df.describe())

#print(df['Age'][0:10])
#print(df.Age[0:10])
#print(df.Age.mean())

#print(df[['Sex', 'Pclass', 'Age']][0:10])

#print(df[df.Age > 60][['Sex', 'Pclass', 'Age', 'Survived']])

#print(df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']])

#df['Age'].hist()
#df['Age'].dropna().hist(bins=16, range=(0,80), alpha=0.5) # Dropped NaN
#plt.show()

#df['Gender'] = 4
#print(df.head(5))
#df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
#print(df.head(5))
df['Gender'] = df['Sex'].map({'female':0, 'male': 1}).astype(int)
#print(df.head(5))


# median ages of gender and class
median_ages = np.zeros((2,3))
print(median_ages)

for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = df[(df.Gender == i) & (df.Pclass == j+1)] \
				['Age'].dropna().median()
print(median_ages)


# Fill age with median_ages
df['AgeFill'] = df['Age']

for i in range(0,2):
	for j in range(0,3):
		df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
				'AgeFill'] = median_ages[i,j]

print(df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10))

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
print(df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill','AgeIsNull']].head(10))

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

# Drop useless column
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

df = df.drop(['Age'],axis=1)

# Drop rows with missing values
df = df.dropna()

# Covert df to a Numpy array for Machine learning
train_data = df.values
print(train_data)
