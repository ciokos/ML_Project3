import numpy as np

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T


def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:
            conf = o.confidence
            supp = r.support
            x = ", ".join(list(o.items_base))
            y = ", ".join(list(o.items_add))
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)" % (x, y, supp, conf))
            frules.append((x, y))
    return frules
