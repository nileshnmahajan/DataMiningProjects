from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from time import time
import pickle
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


svd = TruncatedSVD(n_components=10, algorithm='randomized')
reduced_X = svd.fit_transform(trans_X)

print "Type of reduced X:", type(reduced_X)

print "Shape after dimensionality reduction:", reduced_X.shape

# class KMeans():
#     def set_clusters(self, trans_X):
#         cluster_comp = []
#         clusters = np.random.randint(0, trans_X.shape[0], 7)
#         for c in clusters:
#            cluster_comp.append(trans_X[c])
#         print cluster_comp
#         print type(cluster_comp[0])


centroids = reduced_X[np.random.choice(reduced_X.shape[0], 7)]
print "See if we got some centroids: "
print centroids

print "Size of centroids selection", centroids.shape

print "Type of centroids", type(centroids)



# if __name__ == '__main__':
#     km = KMeans()
#     km.set_clusters(trans_X)


print "Elapsed time: ", time() - start
