import csv
import numpy as np
# from sklearn.preprocessing import OneHotEncoder

datasetpath = "Bike-Sharing-Dataset"

file = open(datasetpath + "/day.csv")
daydata = csv.reader(file, delimiter=',')

next(daydata)
X = []
for row in daydata:    X.append(row[2:])

X = np.array(X)

XX = np.array(X)

# data for clustering
XX = np.array(X)
XX = np.delete(XX, 1, 1)
XX = np.delete(XX, range(2, 5), 1)
XX = np.delete(XX, range(7, 9), 1)
XX = np.delete(XX, 4, 1)

attribute_names = ['month', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt']
classNames = ['winter', 'spring', 'summer', 'fall']

# XX = np.delete(XX, range(1, 5), 1)
# XX = np.delete(XX, 4, 1)
# XX = np.delete(XX, range(6, 8), 1)
#
# cat_idx = [0, 2]
#
# enc = OneHotEncoder(sparse=False, categorical_features=cat_idx)
# X = enc.fit_transform(XX)
#
# attribute_names = ['winter', 'spring', 'summer', 'fall', 'workingday', 'cloudy', 'rainy', 'clear',
#                    'temp', 'hum', 'windspeed', 'cnt']
