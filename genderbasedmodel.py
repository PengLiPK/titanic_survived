import csv as csv
import numpy as np


# Read data
csv_file_object = csv.reader(open('./train.csv'))

data = []
rownum = 0
for row in csv_file_object:
	rownum += 1
	if rownum > 1:
		data.append(row)
data = np.array(data)

# Training gender based model
number_passengers = np.size(data[:,1].astype(np.float))
number_survived = np.sum(data[:,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[:,4] == "female"
men_only_stats = data[:,4] != "female"

women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

proportion_women_survived = np.sum(women_onboard)/np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard)/np.size(men_onboard)


print("Proportion of women who survived is %s" % proportion_women_survived)
print("Proportion of men who survived is %s" % proportion_men_survived)


# Testing gender based model
test_file_object = csv.reader(open('./test.csv'))
next(test_file_object)

prediction_file = open('genderbasedmodel.csv','w+')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

for row in test_file_object:
	if row[3] == 'female':
		prediction_file_object.writerow([row[0],'1'])
	else:
		prediction_file_object.writerow([row[0],'0'])
prediction_file.close()
