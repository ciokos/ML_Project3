from sklearn.decomposition import PCA
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture
from data_preparation import XX, y, attribute_names, classNames
from sklearn.preprocessing import normalize

X = XX
X = normalize(X, axis=0, norm='max')
print(X[0])

pca = PCA(n_components=2)
X = pca.fit_transform(X)

print(X[0])

N, M = X.shape
# C = len(classNames)
# Number of clusters
K = 10
cov_type = 'full'
# type of covariance, you can try out 'diag' as well
reps = 10
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, reg_covar=0.001).fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K, M, M])

    count = 0
    for elem in covs:
        temp_m = np.zeros([M, M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
figure(figsize=(14, 9))
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
show()
