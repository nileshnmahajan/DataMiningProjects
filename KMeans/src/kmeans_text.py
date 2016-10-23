#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
from collections import defaultdict
import operator
import numpy as np
import math
from feature_selec import reduced_featurespace
import pandas as pd
from sklearn.metrics import silhouette_score

submission_path= '../predictions/kmeans_3.txt'


def load_data():
    features = reduced_featurespace
    return features


def euc_distance(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.sqrt(dot(d, d))


def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v,w))


def cosine(f1, f2):
    prod = dot(f1, f2)
    len1 = math.sqrt(dot(f1, f2))
    len2 = math.sqrt(dot(f1, f2))
    return prod / (len1 * len2)


def mean(feats):
    return tuple(np.mean(feats, axis=0))


def assign_cluster(centroids):
    new_centroids = defaultdict(list)
    for cx in centroids:
        for x in centroids[cx]:
            lease_dist = min(centroids, key=lambda c: euc_distance(x, c))
            new_centroids[lease_dist] += [x]
    return new_centroids


def update(centroids):
    new_centroids = {}
    for c in centroids:
        new_centroids[mean(centroids[c])] = centroids[c]
    return new_centroids


def kmeans(features, k, n_iter=100):
    centroids = dict((c, [c]) for c in features[:k])
    centroids[features[k-1]] += features[k:]
    for i in range(n_iter):
        new_centroids = assign_cluster(centroids)
        new_centroids = update(new_centroids)
        if centroids == new_centroids:
            print "KMeans has converged with n_iter=", n_iter
            break
        else:
            centroids = new_centroids
    return centroids


def predict():
    try:
        data = load_data()
    except:
        print "Could not load data. Exiting..."
        sys.exit(1)
    features = data
    clusters = kmeans(features, 7)
    count = 1
    with open(submission_path, 'w') as out:
        for c in clusters:
            print "Size of cluster: ", len(clusters[c])
            for x in clusters[c]:
                out.write(str(features.index(x)) + "," +str(count) + "\n")
            count += 1


def create_submission_file(path):
    solution = pd.read_csv(path, sep=',', names=['indices', 'cluster'])
    solution.indices = solution.indices.astype(int)
    solution.cluster = solution.cluster.astype(int)
    solution = solution.sort_values(by='indices', ascending=True)
    solution.to_csv(submission_path, columns=['cluster'], index=False, header=False)
    print "Silhouette score", silhouette_score(np.asarray(load_data()), solution.cluster, metric='euclidean')


if __name__ == "__main__":
    predict()
    create_submission_file(submission_path)

