#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
from collections import defaultdict
import random
import operator
import numpy as np
import math
from feature_selec import reduced_featurespace
import pandas as pd


def load_data():
    features = reduced_featurespace
    return features


def euc_distance(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.sqrt(np.dot(d, d))


def dot_product(f1, f2):
    return sum(map(operator.mul, f1, f2))


def cosine(f1, f2):
    prod = dot_product(f1, f2)
    len1 = math.sqrt(dot_product(f1, f1))
    len2 = math.sqrt(dot_product(f2, f2))
    return prod / (len1 * len2)


def mean(feats):
    return tuple(np.mean(feats, axis=0))


def assign(centroids):
    new_centroids = defaultdict(list)
    for cx in centroids:
        for x in centroids[cx]:
            best = min(centroids, key=lambda c: 1-cosine(x, c))
            # best = min(centroids, key=lambda c: euc_distance(x, c))
            new_centroids[best] += [x]
    return new_centroids


def update(centroids):
    new_centroids = {}
    for c in centroids:
        new_centroids[mean(centroids[c])] = centroids[c]
    return new_centroids


def kmeans(features, k, maxiter=100):
    centroids = dict((c, [c]) for c in features[:k])
    print "Centroids step 1:", centroids
    centroids[features[k-1]] += features[k:]
    print "Centroids step 2:", centroids
    for i in range(maxiter):
        new_centroids = assign(centroids)
        new_centroids = update((new_centroids))
        if centroids == new_centroids:
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
    with open('../predictions/kmeans_2.txt', 'w') as out:
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
    solution.to_csv('../predictions/kmeans_sub_2.txt', columns=['cluster'], index=False, header=False)


if __name__ == "__main__":
    predict()
    create_submission_file('../predictions/kmeans_1.txt')

