from explore import genre_train
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean').fit(genre_train)
distances, indices = nbrs.kneighbors(genre_train)

print indices[0]

np.savetxt('knn_indices.txt', distances, fmt='%i')