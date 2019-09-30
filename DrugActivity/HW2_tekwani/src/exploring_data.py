import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix


df_train = pd.read_csv('../data/train.csv', sep='\t', index_col=False, header=None,
                       names=['Active', 'Structure'])

df_test = pd.read_csv('../data/test.csv', index_col=False, header=None, names=['Structure'])


def save_sparse_matrix(array):
    np.savez('features', data=array.data, indices=array.indices, indptr=
             array.indptr, shape=array.shape)


print(df_test)

vec = CountVectorizer(binary=True, vocabulary=[str(i) for i in range(100000)])
X_train = vec.fit_transform(df_train['Structure'])
X_test = vec.fit_transform(df_test['Structure'])
Y_train = df_train['Active']

train_labels = Y_train.values

#
# print("Are train labels now an array?" , type(train_labels))
#
# print("Type of train class labels", type(Y))
#
Y_train.to_csv('baseline_train.txt', index=False)
#
#
print("X_train ", X_train)
print("X_test ", X_test)


featurespace_dense_train = X_train.toarray()
featurespace_dense_test = X_test.toarray()


selector = VarianceThreshold()
vt = selector.fit_transform(X_train)

t = time()
save_sparse_matrix(vt)
print("Variance threshold calculated in ", (time() - t))

