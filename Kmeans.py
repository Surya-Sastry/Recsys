# To prevent Subscript for class "list" will generate runtime exception; enclose type annotation in quotes
from __future__ import annotations
import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy
from sklearn.preprocessing import MinMaxScaler


def eucledian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))  # vector based


def minkowski_distance(x1, x2, p):
    from scipy.spatial import distance
    return distance.minkowski(x1, x2, p)


class Kmeans:
    # K = number of clusters
    # iter = number of iterations
    def __init__(self, K, iter, p, random_state=42):
        self.K = K
        self.iter = iter
        self.p = p
        # Initialising empty K clusters -> storing indicies
        self.clusters = [[] for _ in range(self.K)]
        # storing centroids
        self.centroids = []
        self.inertia = 0.0
        self.random_state = random_state
        np.random.seed(random_state)

    def plus_plus(self, ds: np.array, k, random_state=1000):
        """
        Create cluster centroids using the k-means++ algorithm.
        Parameters
        ----------
        ds : numpy array
            The dataset to be used for centroid initialization.
        k : int
            The desired number of clusters for which centroids are required.
        Returns
        -------
        centroids : numpy array
            Collection of k centroids as a numpy array.
        Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
        """

        np.random.seed(random_state)
        centroids = [ds[0]]

        for _ in range(1, k):
            dist_sq = np.array([min([np.inner(c-x, c-x)
                               for c in centroids]) for x in ds])
            probs = dist_sq/dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(ds[i])

        return np.array(centroids)

    def predict(self, X, choice=0):  # no fit required for unsupervised learning models
        self.X = X
        self.n_samples, self.n_features = X.shape  # numpy N-d array

        # initialise centroids
        if choice == 0:
            np.random.seed(self.random_state)
            # row , column full with zero
            centroids = np.zeros((self.K, self.n_features))
            for k in range(self.K):  # iterations of
                # random centroids
                centroid = X[np.random.choice(range(self.n_samples))]
                centroids[k] = centroid
            self.centroids = centroids
        elif choice == 1:
            self.centroids = self.plus_plus(X, self.K)

        elif choice == 2:
            self.centroids = self.naive_sharding(X, self.K)

        # optimise
        for _ in range(self.iter):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            # update centroids
            centroids_old = self.centroids  # for convergence test
            self.centroids = self._get_centroids(self.clusters)
            # print(self.centroids)
            # get cetnroids assign mean value of cluster to the centroid
            # check for convergence
            if self._isConverged(centroids_old, self.centroids):
                break

            # break

        # classify the samples based on index of cluster
        labels = self.getClusterLabels(self.clusters)
        self.calculateInertia(X, labels)
        return labels, self.centroids
        # return cluster labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            # find the closest centroid to classify, then put it in the specific cluster
            # here cluster is an list of list
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def function(self, point):
        return minkowski_distance(self.sample, point, self.p)

    def _closest_centroid(self, sample, centroids):
        # distance metric
        distances = [minkowski_distance(sample, point, self.p)
                     for point in centroids]
        # argmin for required parameters
        closest_idx = np.argmin(distances)
        # self.sample=sample
        # closest_idx = scipy.optimize.minimize (fun=function ,x0=np.asarray(centroids))
        return closest_idx

    def _get_centroids(self, clusters):
        # Dimension = N as N = number of features
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            # calling mean on a specific current cluster as X is array of various clusters
            if (len(cluster) != 0):
                cluster_mean = np.mean(self.X[cluster], axis=0)
                # set mean as newer centroids
                centroids[cluster_idx] = cluster_mean
        return centroids

    def _isConverged(self, centroids_old, centroids):
        distances = [minkowski_distance(
            centroids_old[i], centroids[i], self.p) for i in range(self.K)]
        # no more change occured in 2 iteration so converges
        return sum(distances) == 0

    def getClusterLabels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def calculateInertia(self, datapoints: list[int], labels: list[any]) -> None:
        labels = labels.astype(int)
        for idx, pts in enumerate(datapoints):
            self.inertia += minkowski_distance(
                self.centroids[labels[idx]], pts, 2)**2

    def predictPoint(self, datapoint):
        distances = []
        for idx, i in enumerate(self.centroids):
            if sum(i) == 0:
                continue
            else:
                distances.append(
                    [minkowski_distance(datapoint, i, self.p), idx])
        distances.sort()
        return distances[0][1]