from outliers_detection import gkd, knnd, ard
import numpy as np



gkd = set(list(gkd[:20]))
knnd = set(list(knnd[:20]))
ard = set(list(ard[:20]))

com12 = gkd.intersection(knnd)
com13 = gkd.intersection(ard)
com23 = knnd.intersection(ard)
com123 = com12.intersection(ard)

print("GKD + KNN: {0}".format(com12))
print("GKD + ARD: {0}".format(com13))
print("KNN + ARD: {0}".format(com23))
print("GKD + KNN + ARD: {0}".format(com123))
