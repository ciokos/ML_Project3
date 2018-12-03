from data_preparation import attribute_names2, X2
from similarity import binarize2
import numpy as np
import utility
from apyori import apriori


b_idx = [8, 9, 10, 11]
X2bin = X2[:, b_idx]

attr2bin = attribute_names2[-4:]

Xbin, attribute_names_bin = binarize2(X2bin, attr2bin)

X = np.zeros((731, 16))
X[:, :7] = X2[:, :7]
X[:, 8:] = Xbin

attribute_names = attribute_names2[:8] + attribute_names_bin

T = utility.mat2transactions(X, labels=attribute_names)
rules = apriori(T, min_support=0.35, min_confidence=.6)
print("Rules:")
utility.print_apriori_rules(rules)


print("Frequent Itemsets:")
fis = apriori(T, min_support=0.35, min_confidence=0)
utility.print_apriori_rules(fis)


