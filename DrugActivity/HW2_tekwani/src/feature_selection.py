import numpy as np
from scipy.sparse import csr_matrix
from exploring_data import X, selector, featurespace_dense, Y
from collections import Counter
import pandas as pd


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

F = load_sparse_csr('features.npz')
v = selector.fit(X)

var = v.variances_

# print "Features with highest variance", np.argsort(var)[-10:]

thresh_0 = X[:, var > 0.11]

# print "thresh0: ", thresh_0
#
# # tells us how many features are left
# print thresh_0.shape

#feature numbers for the ones that are left
idx = np.where(v.variances_ > 0.08)[0]

# print "Feature numbers with highest variance" , idx
# print idx.shape
# print "no of features passing through current variance filter", len(idx)

# print type(idx)

# print "The features which are above the threshold: ", selector.get_support(indices=True)
# #No of features being used
# used = str(Counter(selector.get_support())[True])
# discarded = str(Counter(selector.get_support())[False])
# print "Using: ", used
# print "Discarded: ", discarded


df_reduced = pd.DataFrame(np.nan, index=range(800), columns=idx)


def get_value_featurespace(row, column):
    return featurespace_dense[row, column]


def populate_df_reduced(row, col):
    df_reduced.xs(row)[col] = get_value_featurespace(row, col)


for i in range(800):
    for j in idx:
        populate_df_reduced(i, j)


result = pd.concat([df_reduced, Y], axis=1)

print result







