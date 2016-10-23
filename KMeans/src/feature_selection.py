from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from time import time

data = []

with open('../data/input.mat', 'r') as file:
    for i, line in enumerate(file):
        l = line.split()
        d = dict([(k, v) for k, v in zip(l[::2], l[1::2])])
        data.append(d)

v = DictVectorizer(sparse=True, dtype=float)
X = v.fit_transform(data)

start = time()


tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
trans_X = tfidf.fit_transform(X)

print "Shape of original featureset: ", trans_X.shape

pca = PCA(n_components=10)
pca.fit(trans_X.toarray())

print "Loaded model in ", time() - start

reduced_X = pca.fit_transform(trans_X.toarray())

X_arr = tuple(map(tuple, reduced_X))
reduced_featurespace = list(X_arr)

print "Explained variance", pca.explained_variance_
print "Components", pca.components_
print "Elapsed time: ", time() - start