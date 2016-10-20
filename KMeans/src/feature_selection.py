from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from time import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

# train_data = pd.read_csv('../dataput.csv', header=None)
# train_data['Spaces'] = train_data.applymap(lambda x: str.count(x, ' '))
# train_data['Words'] = train_data['Spaces']/2

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


print "Transformed featurespace: "
print trans_X

print "Elapsed time: ", time() - start


# pca = PCA(n_components=2)
# reduced_X = pca.fit_transform(X.toarray())

svd = TruncatedSVD(n_components=10, algorithm='randomized')
reduced_X = svd.fit_transform(trans_X)

kmeans = KMeans(init='k-means++', n_clusters=7, max_iter=100)
kmeans.fit(reduced_X)

labels = kmeans.predict(reduced_X)

np.savetxt('../predictions/kmeans_svd_10_norm.txt', labels, fmt='%i')


svdpkl = pickle.dumps(svd)


print "Explained variance : "

try:

    print svd.explained_variance_
    print svd.components_

except:
    pass

print "Time elapsed: ", time() - start
