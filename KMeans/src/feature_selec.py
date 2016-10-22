from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
# from sklearn.cluster import KMeans
from time import time
import sys
from sklearn.externals import joblib
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = []

with open('../data/input.mat', 'r') as file:
    for i, line in enumerate(file):
        l = line.split()
        d = dict([(k, v) for k, v in zip(l[::2], l[1::2])])
        data.append(d)

v = DictVectorizer(sparse=True, dtype=float)
X = v.fit_transform(data)

start = time()


tfidf = TfidfTransformer(norm='l2', use_idf=True)
trans_X = tfidf.fit_transform(X)

# svd = TruncatedSVD(n_components=10, algorithm='randomized')
# reduced_X = svd.fit_transform(trans_X)

# pca = PCA(n_components=10)
# pca.fit(trans_X.toarray())

# joblib.dump(pca, 'pca_10.pkl')

pca = joblib.load('pca_10.pkl')

reduced_X = pca.fit_transform(trans_X.toarray())

try:
    X_arr = tuple(map(tuple, reduced_X))

    reduced_featurespace = list(X_arr)

except:
    sys.exit(1)

# print len(reduced_featurespace)
# print reduced_featurespace

print "Elapsed time: ", time() - start


try:

    # print svd.explained_variance_
    # print svd.components_

    print "Explained variance", pca.explained_variance_
    print "Components", pca.components_

except:
    pass

