import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix


df_train = pd.read_csv('../data/train.csv', sep='\t', index_col=False, header=None,
                       names=['Active', 'Structure'])


def save_sparse_matrix(array):
    np.savez('features', data=array.data, indices=array.indices, indptr=
             array.indptr, shape=array.shape)


vec = CountVectorizer(binary=True, vocabulary=[str(i) for i in range(100000)])
X = vec.fit_transform(df_train['Structure'])
Y = df_train['Active']

print "X: ", X

featurespace_dense = X.toarray()

selector = VarianceThreshold()
vt = selector.fit_transform(X)

t = time()
save_sparse_matrix(vt)
print "Variance threshold calculated in ", (time() -t)

