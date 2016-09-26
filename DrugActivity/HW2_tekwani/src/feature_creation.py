import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix


df_train = pd.read_csv('../data/train.csv', sep='\t', index_col=False, header=None,
                       names=['Active', 'Structure'])

vec = CountVectorizer(binary=True, vocabulary=[str(i) for i in range(100000)])
X = vec.fit_transform(df_train['Structure'])
y = df_train['Active'].values

featurespace_dense_X = X.toarray()

selector = VarianceThreshold()
vt = selector.fit_transform(X)

