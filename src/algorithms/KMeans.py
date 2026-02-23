import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # store the indices of each point that belongs to each clusters
        self.clusters = [[] for _ in range(self.K)]

        self.centroids = None


    def predict(self, X: np.ndarray):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # init random point as centroid (without replacement)
        random_sample_indices = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_indices]

        # reassign centroids iteratively
        for _ in range(self.max_iters):
            # assign points to nearest clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroid based on clusters
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(old_centroids, self.centroids):
                break

            if self.plot_steps:
                self.plot()
        
        return self._get_cluster_labels(self.clusters)

    
    def _create_clusters(self, centroids):
        # assign points to nearest clusters

        clusters = [[] for _ in range(self.K)]

        for idx, sample in enumerate(self.X):
            nearest_centroid = self._nearest_centroid(sample, centroids)
            clusters[nearest_centroid].append(idx)
        
        return clusters
    
    def _nearest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        nearest_centroid = np.argmin(distances)

        return nearest_centroid

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))

        for idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean

        return centroids
    
    def _is_converged(self, old_centroids, centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.K)]

        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for idx, _ in enumerate(cluster):
                labels[idx] = cluster_idx
        
        return labels