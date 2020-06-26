import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import copy
from optimize.functions import MSTER, LCR, loss, grad, H
from matplotlib import pyplot as plt

class datasets():
	def __init__(self, distance = 5, default_ratio = 10, seed = 1600, n = 2000, num_dimensions = 2, imbalance_ratio = 10):
		from sklearn import datasets

		self.distance = distance
		self.std_ratio = default_ratio
		self.std = distance / self.std_ratio
		self.seed = seed
		np.random.seed(self.seed)
		self.num_points = n
		self.num_dimensions = num_dimensions
		self.imbalance_ratio = imbalance_ratio

		###dictionary format; dict[k] returns tuple (dataset, labels) where k is number of clusters
		self.balanced_clusters_normal = {}
		self.balanced_clusters_tight = {}
		self.balanced_clusters_loose = {}
		self.balanced_clusters_doubled = {}
		self.imbalanced_clusters = {}
		self.outlier_clusters = {}
		self.simplexes = {}
		self.polyhedra = {}

		###imbalanced clusters; ONE pair of adjacent clusters is imbalanced with one having 4x the number of the other. All other clusters are unaffected
		self.imbalanced_point_distributions = {}
		for i in range(2,7):
			distribution = []
			distribution.append(int(self.imbalance_ratio * 2 * self.num_points / ((self.imbalance_ratio+1) * i)))
			distribution.append(int(2 * self.num_points / ((self.imbalance_ratio+1) * i)))
			for j in range(2, i):
				distribution.append(int(self.num_points / i))
			self.imbalanced_point_distributions[i] = distribution

		###2 clusters is just separated along x axis
		k = 2
		arr = np.zeros((k ,2))
		arr[0,0:2] = [self.distance/2, 0]
		arr[1,0:2] = [-self.distance/2, 0]
		
		self.balanced_clusters_normal[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.imbalanced_clusters[k] = list(datasets.make_blobs(n_samples=self.imbalanced_point_distributions[k], n_features=k, centers = arr, cluster_std=self.std))

		###3 clusters is equilateral triangle
		k=3
		arr = np.zeros((k ,2))
		arr[0,0:2] = [0, np.sqrt(3)*self.distance/4]
		arr[1,0:2] = [-self.distance/2, -np.sqrt(3)*self.distance/4]
		arr[2,0:2] = [self.distance/2, -np.sqrt(3)*self.distance/4]

		self.balanced_clusters_normal[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.imbalanced_clusters[k] = list(datasets.make_blobs(n_samples=self.imbalanced_point_distributions[k], n_features=k, centers = arr, cluster_std=self.std))

		###4 clusters is square
		k=4
		arr = np.zeros((k ,2))
		arr[0,0:2] = [-self.distance/2, -self.distance/2]
		arr[1,0:2] = [-self.distance/2, self.distance/2]
		arr[2,0:2] = [self.distance/2, -self.distance/2]
		arr[3,0:2] = [self.distance/2, self.distance/2]

		self.balanced_clusters_normal[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.imbalanced_clusters[k] = list(datasets.make_blobs(n_samples=self.imbalanced_point_distributions[k], n_features=k, centers = arr, cluster_std=self.std))

		###5 clusters is pentagon
		k=5
		arr = np.zeros((k ,2))
		arr[0,0:2] = [0, self.distance]
		arr[1,0:2] = [self.distance * np.cos(np.pi/10), self.distance * np.sin(np.pi/10)]
		arr[2,0:2] = [self.distance * np.cos(-np.pi * 0.3), self.distance * np.sin(-np.pi * 0.3)]
		arr[3,0:2] = [-self.distance * np.cos(np.pi/10), self.distance * np.sin(np.pi/10)]
		arr[4,0:2] = [-self.distance * np.cos(-np.pi * 0.3), self.distance * np.sin(-np.pi * 0.3)]

		self.balanced_clusters_normal[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.imbalanced_clusters[k] = list(datasets.make_blobs(n_samples=self.imbalanced_point_distributions[k], n_features=k, centers = arr, cluster_std=self.std))

		###6 clusters is hexagon
		k=6
		arr = np.zeros((k ,2))
		arr[0,0:2] = [0, self.distance]
		arr[1,0:2] = [self.distance * np.sqrt(3)/2, self.distance * 0.5]
		arr[2,0:2] = [self.distance * np.sqrt(3)/2, -self.distance * 0.5]
		arr[3,0:2] = [0, -self.distance]
		arr[4,0:2] = [-self.distance * np.sqrt(3)/2, self.distance * 0.5]
		arr[5,0:2] = [-self.distance * np.sqrt(3)/2, -self.distance * 0.5]

		self.balanced_clusters_normal[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[k] = list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))
		self.imbalanced_clusters[k] = list(datasets.make_blobs(n_samples=self.imbalanced_point_distributions[k], n_features=k, centers = arr, cluster_std=self.std))


		###outlier clusters
		#TODO IDK HOW SHUD ADD OUTLIERS

		###simplex
		self.max_dimension = 10


		###polyhedron
		self.max_polyhedron = 6


		###normalise everything by dividing by trace of covariance matrix
		for k in range(2, 7):
			self.balanced_clusters_normal[k][0] = 100 * np.mat(np.mat(self.balanced_clusters_normal[k][0])) / (np.trace(np.mat(self.balanced_clusters_normal[k][0]) * np.mat(self.balanced_clusters_normal[k][0]).T))
			self.balanced_clusters_tight[k][0] = 100 * np.mat(np.mat(self.balanced_clusters_tight[k][0])) / (np.trace(np.mat(self.balanced_clusters_tight[k][0]) * np.mat(self.balanced_clusters_tight[k][0]).T))
			self.balanced_clusters_loose[k][0] = 100 * np.mat(np.mat(self.balanced_clusters_loose[k][0])) / (np.trace(np.mat(self.balanced_clusters_loose[k][0]) * np.mat(self.balanced_clusters_loose[k][0]).T))
			self.balanced_clusters_doubled[k][0] = 200 * np.mat(np.mat(self.balanced_clusters_doubled[k][0])) / (np.trace(np.mat(self.balanced_clusters_doubled[k][0]) * np.mat(self.balanced_clusters_doubled[k][0]).T))

	def get_datasets(self):
		return (
			self.balanced_clusters_normal,
			self.balanced_clusters_tight,
			self.balanced_clusters_loose,
			self.balanced_clusters_doubled,
			self.imbalanced_clusters,
			self.outlier_clusters,
			self.simplexes,
			self.polyhedra
		)


if __name__ == "__main__":

	balanced_clusters_normal, balanced_clusters_tight, balanced_clusters_loose, balanced_clusters_doubled, imbalanced_clusters, outlier_clusters, simplexes, polyhedra = datasets().get_datasets()

	k=6

	plt.figure(1)
	plt.scatter(x = np.array(balanced_clusters_normal[k][0][:,0]), y = np.array(balanced_clusters_normal[k][0][:,1]))
	plt.axis('equal')

	plt.figure(2)
	plt.scatter(x = np.array(balanced_clusters_tight[k][0][:,0]), y = np.array(balanced_clusters_tight[k][0][:,1]))
	plt.axis('equal')

	plt.figure(3)
	plt.scatter(x = np.array(balanced_clusters_loose[k][0][:,0]), y = np.array(balanced_clusters_loose[k][0][:,1]))
	plt.axis('equal')

	plt.figure(4)
	plt.scatter(x = np.array(balanced_clusters_doubled[k][0][:,0]), y = np.array(balanced_clusters_doubled[k][0][:,1]))
	plt.axis('equal')

	plt.figure(5)
	plt.scatter(x = np.array(imbalanced_clusters[k][0][:,0]), y = np.array(imbalanced_clusters[k][0][:,1]))
	plt.axis('equal')

	plt.show()