# exercise 11.1.1
from matplotlib.pyplot import figure, plot, legend, xlabel, show, title, ylim
import numpy as np
from sklearn.mixture import GaussianMixture
from data_preparation import XX, y, attribute_names, classNames
from sklearn import model_selection
from sklearn.preprocessing import normalize
from toolbox_02450 import clusterval

# data
X = XX
X = normalize(X, axis=0, norm='max')

N, M = X.shape
# C = len(classNames)

# Range of K's to try
KRange = range(9, 11)
T = len(KRange)

covar_type = 'full'  # you can try out 'diag' as well
reps = 10  # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=8, shuffle=True)
record = float("inf")
chosen_K = -1
centroids = []

# Allocate variables:


for t, K in enumerate(KRange):
    print('Fitting model for K={0}'.format(K))

    # Fit Gaussian mixture model
    # gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)
    #
    # # Get BIC and AIC
    # BIC[t, ] = gmm.bic(X)
    # AIC[t, ] = gmm.aic(X)

    # For each crossvalidation fold
    for train_index, test_index in CV.split(X):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        X_test = X[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps, reg_covar=0.0001).fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score_samples(X_test).sum()

    if CVE[t] < record:
        record = CVE[t]
        chosen_K = K
        centroids = gmm.means_
        cls = gmm.predict(X)
        Rand, Jaccard, NMI = clusterval(y, cls)


print(Rand, Jaccard, NMI)
print(chosen_K)
# Plot results
figure(1)
# plot(KRange, BIC, '-*b')
# plot(KRange, AIC, '-xr')
plot(KRange, 2 * CVE, '-ok')
legend(['Crossvalidation'])
# legend(['BIC', 'AIC', 'Crossvalidation'])

xlabel('K')
show()

# figure(2)
# title('Cluster validity')
# plot(np.arange(K) + 1, Rand)
# plot(np.arange(K) + 1, Jaccard)
# plot(np.arange(K) + 1, NMI)
# ylim(-2, 1.1)
# legend(['Rand', 'Jaccard', 'NMI'], loc=4)
# show()
