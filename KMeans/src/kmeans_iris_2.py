#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.metrics import silhouette_score
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import math
import random

DATA = '../data/iris.csv'
submission_path= '../predictions/kmeans_iris_3.txt'


def load_data():
    data = [l.strip() for l in open(DATA) if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
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


def mean(features):
    return tuple(np.mean(features, axis=0))


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
    # centroids = dict((features.index(c), c) for c in features[:k])
    feature_sample = random.sample(features, k)
    print "Feature sample", feature_sample
    centroids = dict((c, [c]) for c in feature_sample)
    # centroids = dict((c, [c]) for c in features[:k])
    print centroids
    # temp = random.sample(features, 3)
    # centroids = dict((temp, [temp]) for temp in features[:k])
    # print centroids
    centroids[feature_sample[k-1]] += features[k:]
    print "here", centroids
    for i in range(n_iter):
        new_centroids = assign_cluster(centroids)
        new_centroids = update(new_centroids)
        if centroids == new_centroids:
            print "KMeans has converged with n_iter=", n_iter
            break
        else:
            centroids = new_centroids
    return centroids


def predict_clusters():
    try:
        features = load_data()
    except:
        print "Could not load data. Exiting...."
        sys.exit(1)
    clusters = kmeans(features, 3)
    count = 1
    with open(submission_path, 'w') as out:
        for c in clusters:
            print "Size of cluster ", count, len(clusters[c])
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
    predict_clusters()
    create_submission_file(submission_path)


