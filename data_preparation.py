import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder

datasetpath = "Bike-Sharing-Dataset"

file = open(datasetpath + "/day.csv")
daydata = csv.reader(file, delimiter=',')

row1 = next(daydata)
raw_attribute_names = row1[2:]
X = []
for row in daydata:    X.append(row[2:])

X = np.array(X)

XX = np.array(X)

# data for clustering
XX = np.array(X)
y = XX[:, 0]
XX = np.delete(XX, [0, 1, 2, 3, 4, 5, 6], 1)
# cat_idx = [0]
# #
# enc = OneHotEncoder(sparse=False, categorical_features=cat_idx)
# XX = enc.fit_transform(XX)

attribute_names = ['month', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt']
classNames = ['winter', 'spring', 'summer', 'fall']


#data for association mining
X2 = np.array(X)
X2 = np.delete(X2, range(1, 5), 1)
X2 = np.delete(X2, 4, 1)
X2 = np.delete(X2, range(6,8), 1)

cat_idx = [0, 2]

enc = OneHotEncoder(sparse=False, categorical_features=cat_idx)
X2 = enc.fit_transform(X2)

attribute_names2 = ['winter', 'spring', 'summer', 'fall', 'workingday', 'cloudy', 'rainy', 'clear',
                   'temp', 'hum', 'windspeed', 'cnt']
