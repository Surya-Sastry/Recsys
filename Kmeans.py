import numpy as np
import pandas as pd

def eucledian_distance(x1, x2):
	return np.sqrt(np.sum((x1-x2)**2))

def minkowski_distance(x1, x2, p):
	return np.power(np.sum(np.abs(x1-x2)**p),1/p)

def manhattan_distance(x1,x2):
	return np.sum(np.abs(x1-x2))

def distance_func(x1,x2,p,choice):
	match choice:
		case 'Eucledian':
			return eucledian_distance(x1,x2)
		case 'Minkowski':
			return minkowski_distance(x1,x2,p)
		case 'Manhattan':
			return manhattan_distance(x1,x2)
		case _:
			return eucledian_distance(x1,x2)

class Kmeans:
	def __init__(self, K, iter, p, dist_choice, random_state=42):
		self.K = K
		self.iter = iter
		self.p = p
		self.clusters = [[] for _ in range(self.K)]
		self.centroids = []
		self.inertia = 0.0
		self.random_state = random_state
		self.dist_choice = dist_choice
		np.random.seed(random_state)

	def plusplus(self, ds: np.array, k, random_state=1000):
		np.random.seed(random_state)
		centroids = [ds[0]]

		for _ in range(1, k):
			dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in ds])
			probs = dist_sq/dist_sq.sum()
			cumulative_probs = probs.cumsum()
			r = np.random.rand()
			for j, p in enumerate(cumulative_probs):
				if r < p:
					i = j
					break
			centroids.append(ds[i])
		return np.array(centroids)

	def naive_sharding(self,ds:np.array, k):
		n = np.shape(ds)[1]
		m = np.shape(ds)[0]
		centroids = np.mat(np.zeros((k,n)))
		composite = np.mat(np.sum(ds, axis=1))
		ds = np.append(composite.T, ds, axis=1)
		ds.sort(axis=0)
		step = floor(m/k)
		def _get_mean(sums, step):
			return sums/step
		vmean = np.vectorize(_get_mean)
		for j in range(k):
			if j == k-1:
				centroids[j:] = vmean(np.sum(ds[j*step:,1:], axis=0), step)
			else:
				centroids[j:] = vmean(np.sum(ds[j*step:(j+1)*step,1:], axis=0), step)
		return np.array(centroids)

	def predict(self, X, choice=0):
		self.X = X
		self.n_samples, self.n_features = X.shape

		match choice:
			case 0:
				np.random.seed(self.random_state)
				centroids = np.zeros((self.K, self.n_features))
				for k in range(self.K):
					centroid = X[np.random.choice(range(self.n_samples))]
					centroids[k] = centroid
				self.centroids = centroids
			case 1:
				self.centroids = self.plusplus(X, self.K)

			case 2:
				self.centroids = self.naive_sharding(X, self.K)

		for _ in range(self.iter):
			self.clusters = self._create_clusters(self.centroids)
			centroids_old = self.centroids
			self.centroids = self._get_centroids(self.clusters)
			if self._isConverged(centroids_old, self.centroids):
				break

		labels = self.getClusterLabels(self.clusters)
		self.calculateInertia(X, labels)
		return labels, self.centroids

	def _create_clusters(self, centroids):
		clusters = [[] for _ in range(self.K)]
		for idx, sample in enumerate(self.X):
			centroid_idx = self._closest_centroid(sample, centroids)
			clusters[centroid_idx].append(idx)
		return clusters

	def test_function(self, point):
		return distance_func(self.sample, point, self.p,self.dist_choice)

	def _closest_centroid(self, sample, centroids):
		distances = [distance_func(sample, point, self.p,self.dist_choice) for point in centroids]
		closest_idx = np.argmin(distances)
		return closest_idx

	def _get_centroids(self, clusters):
		centroids = np.zeros((self.K, self.n_features))
		for cluster_idx, cluster in enumerate(clusters):
			if (len(cluster) != 0):
				cluster_mean = np.mean(self.X[cluster], axis=0)
				centroids[cluster_idx] = cluster_mean
		return centroids

	def _isConverged(self, centroids_old, centroids):
		distances = [distance_func(centroids_old[i], centroids[i], self.p,self.dist_choice) for i in range(self.K)]
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
			self.inertia += distance_func(self.centroids[labels[idx]], pts, 2 , self.dist_choice) ** 2

	def predictPoint(self, datapoint):
		distances = []
		for idx, i in enumerate(self.centroids):
			if sum(i) == 0:
				continue
			else:
				distances.append( [distance_func(datapoint, i, self.p), idx] , self.dist_choice)
		distances.sort()
		return distances[0][1]

if __name__ == "__main__":
	Kmeans(5,iter,2,'Eucledian')
	print("Test run")
