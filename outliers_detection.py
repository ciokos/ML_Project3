import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors
from data_preparation import X, raw_attribute_names
from sklearn.preprocessing import normalize

X = np.array(X).astype(float)
X = normalize(X, axis=0)
N, M = np.shape(X)

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=1).max() * (2.0 ** np.arange(-10, 10))
logP = np.zeros(np.size(widths))
for i, w in enumerate(widths):
    print('Fold {:2d}, w={:f}'.format(i, w))
    density, log_density = gausKernelDensity(X, w)
    logP[i] = log_density.sum()

val = logP.max()
ind = logP.argmax()

width = widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1, )

gkd = i

print("----Gausian Kernel density----")
print("Lowest density objects:")
for k in range(20):
    print('Point no.: {0}: density: {1} for data object: {2}'.format(k+1, density[k], i[k]))
# print('Point no.: 731: density: {0} for data object: {1}'.format(density[-1], i[-1]))


# Plot density estimate of outlier score
figure(1)
bar(range(10), density[:10])
title('Gausian Kernel Density estimate')

# Plot possible outliers
# figure(2)
# for k in range(1, 21):
#     subplot(4, 5, k)
#
#     xticks([])
#     yticks([])
#     if k == 3: title('Gaussian Kernel Density: Possible outliers')

### K-neighbors density estimator
# Neighbor to use:
K = 50

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1. / (D.sum(axis=1) / K)

# Sort the scores
i = density.argsort()
density = density[i]

knnd = i
print("----K-neighbors density estimator----")
print("Lowest density objects:")
for k in range(20):
    print('Point no.: {0}: density: {1} for data object: {2}'.format(k+1, density[k], i[k]))
print('Point no.: 731: density: {0} for data object: {1}'.format(density[-1], i[-1]))

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(20), density[:20])
title('KNN density: Outlier score')
# Plot possible outliers
# figure(4)
# for k in range(1, 21):
#     subplot(4, 5, k)
#     imshow(np.reshape(X[i[k], :], (16, 16)).T, cmap=cm.binary)
#     xticks([]);
#     yticks([])
#     if k == 3: title('KNN density: Possible outliers')

### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1. / (D.sum(axis=1) / K)
avg_rel_density = density / (density[i[:, 1:]].sum(axis=1) / K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

ard = i_avg_rel
print("----K-nearest neigbor average relative density----")
print("Lowest density objects:")
for k in range(20):
    print('Point no.: {0}: density: {1} for data object: {2}'.format(k+1, avg_rel_density[k], i_avg_rel[k]))
print('Point no.: 731: density: {0} for data object: {1}'.format(avg_rel_density[-1], i_avg_rel[-1]))

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(20), avg_rel_density[:20])
title('KNN average relative density: Outlier score')
# Plot possible outliers
# figure(6)
# for k in range(1, 21):
#     subplot(4, 5, k)
#     imshow(np.reshape(X[i_avg_rel[k], :], (16, 16)).T, cmap=cm.binary)
#     xticks([]);
#     yticks([])
#     if k == 3: title('KNN average relative density: Possible outliers')
#
# ### Distance to 5'th nearest neighbor outlier score
# K = 5
#
# # Find the k nearest neighbors
# knn = NearestNeighbors(n_neighbors=K).fit(X)
# D, i = knn.kneighbors(X)
#
# # Outlier score
# score = D[:, K - 1]
# # Sort the scores
# i = score.argsort()
# score = score[i[::-1]]
#
# print("Lowest density objects:")
# for k in range(20):
#     print('Point no.: {0}: density: {1} for data object: {2}'.format(k+1, density[k], i[k]))
#
# # Plot k-neighbor estimate of outlier score (distances)
# figure(7)
# bar(range(20), score[:20])
# title('5th neighbor distance: Outlier score')
# # Plot possible outliers
# # figure(8)
# # for k in range(1, 21):
# #     subplot(4, 5, k)
# #     imshow(np.reshape(X[i[k], :], (16, 16)).T, cmap=cm.binary)
# #     xticks([]);
# #     yticks([])
# #     if k == 3: title('5th neighbor distance: Possible outliers')
#
# # Plot random digits (the first 20 in the data set), for comparison
# # figure(9)
# # for k in range(1, 21):
# #     subplot(4, 5, k);
# #     imshow(np.reshape(X[k, :], (16, 16)).T, cmap=cm.binary)
# #     xticks([]);
# #     yticks([])
# #     if k == 3: title('Random digits from data set')
show()
