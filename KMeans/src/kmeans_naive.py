#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
from collections import defaultdict
import random
import operator
import numpy as np
import math


def load_data():
    data = [l.strip() for l in open('../data/iris.txt') if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    return dict(zip(features, labels))


# Function for Euclidean distance
def euc_distance(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.sqrt(np.dot(d, d))


def dot_product(f1, f2):
    return sum(map(operator.mul, f1, f2))


def manhattan(f1, f2):
    d = np.subtract(f1, f2).all()
    return d


#Function for cosine similarity
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
            new_centroids[best] += [x]
    return new_centroids


def update(centroids):
    new_centroids = {}
    for c in centroids:
        new_centroids[mean(centroids[c])] = centroids[c]
    return new_centroids


def kmeans(features, k, maxiter=100):
    centroids = dict((c, [c]) for c in features[:k])
    centroids[features[k-1]] += features[k:]
    for i in range(maxiter):
        new_centroids = assign(centroids)
        new_centroids = update((new_centroids))
        if centroids == new_centroids:
            break
        else:
            # centers = new_centers
            centroids = new_centroids
    return centroids


def counter(alist):
    count = defaultdict(int)
    for x in alist:
        count[x] += 1
    return dict(count)


def predict(seed=10):
    try:
        data = load_data()
    except IOError:
        print ("Data missing. Please download the Iris dataset.")
        sys.exit(1)
    features = data.keys()
    print features
    print type(features)
    random.seed(seed)
    random.shuffle(features)
    clusters = kmeans(features, 3)
    with open('../predictions/kmns_6.txt', 'w') as out:
        for c in clusters:
            # print (counter([data[x] for x in clusters[c]]))
            for t in clusters[c]:

                if data[t] == "Iris-setosa":
                    out.write('1' +'\n')
                if data[t] == "Iris-versicolor":
                    out.write('2' + '\n')
                if data[t] == "Iris-virginica":
                    out.write('3' + '\n')


if __name__ == "__main__":
    predict()