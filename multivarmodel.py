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


# Training with multi-varibles (gender, class, ticket price)

# Add a ceiling to the ticket fare
fare_ceiling = 40
data[data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling-1.0

fare_bracket_size = 10
number_of_price_brackets = int(fare_ceiling/fare_bracket_size)

number_of_classes = len(np.unique(data[0::,2]))

# Initialize the survival table
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in range(number_of_classes):
	for j in range(number_of_price_brackets):

		women_only_stats = data[
				(data[0::,4] == "female")
				&(data[0::,2].astype(np.float) == i+1)
				&(data[0:,9].astype(np.float) >= j*fare_bracket_size)
				&(data[0:,9].astype(np.float) <= (j+1)*fare_bracket_size)
				, 1]
		men_only_stats = data[
				(data[0::,4] == "male")
				&(data[0::,2].astype(np.float) == i+1)
				&(data[0:,9].astype(np.float) >= j*fare_bracket_size)
				&(data[0:,9].astype(np.float) <= (j+1)*fare_bracket_size)
				, 1]

		try:
			survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
			survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
		except Exception as e:
			pass
	
survival_table[ survival_table != survival_table ] = 0

print(survival_table)

survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1

print(survival_table)


# Testing multi-variables model
test_file_object = csv.reader(open('./test.csv'))
next(test_file_object)

pred_f = open('multivarmodel.csv','w+')
pred_f_obj = csv.writer(pred_f)
pred_f_obj.writerow(["PassengerId", "Survived"])

for row in test_file_object:
	if row[3] == 'female':
		i = 0
	else:
		i = 1
	
	j = int(row[1]) - 1

	try:
		bin_fare = float(row[8])/10
	except Exception as e:
		bin_fare = 0

	if(bin_fare >= 4):
		k = 3
	else:
		k = bin_fare
	
	pred_f_obj.writerow([row[0], int(survival_table[i,j,k])])

pred_f.close()
