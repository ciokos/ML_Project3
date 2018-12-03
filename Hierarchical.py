from matplotlib.pyplot import figure, show
from toolbox_02450 import clusterplot, clusterval
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from data_preparation import XX, attribute_names, y, classNames
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import PCA

X = XX
X = normalize(X, axis=0, norm='max')

N, M = X.shape
C = len(classNames)

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 10
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
cls = np.array(cls)
Rand, Jaccard, NMI = clusterval(y, cls)
print(Rand, Jaccard, NMI)
figure(1)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
clusterplot(X, cls.reshape(cls.shape[0], 1), y=y)

# Display dendrogram
max_display_levels = 6
figure(2, figsize=(10, 4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
