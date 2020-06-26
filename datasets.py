import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import copy
from optimize.functions import MSTER, LCR, loss, grad, H
from matplotlib import pyplot as plt

class datasets():
	def __init__(self, distance = 5, default_ratio = 10, seed = 1600, n = 2000, num_dimensions = 2):
		from sklearn import datasets

		self.distance = distance
		self.std_ratio = default_ratio
		self.std = distance / self.std_ratio
		self.seed = seed
		np.random.seed(self.seed)
		self.num_points = n
		self.num_dimensions = num_dimensions

		###dictionary format; dict[k] returns tuple (dataset, labels) where k is number of clusters
		self.balanced_clusters_normal = {}
		self.balanced_clusters_tight = {}
		self.balanced_clusters_loose = {}
		self.balanced_clusters_doubled = {}
		self.imbalanced_clusters = {}
		self.outlier_clusters = {}
		self.simplexes = {}
		self.polyhedra = {}

		###2 clusters is just separated along x axis
		arr = np.zeros((2 ,2))
		arr[0,0:2] = [self.distance/2, 0]
		arr[1,0:2] = [-self.distance/2, 0]
		
		self.balanced_clusters_normal[2] = list(datasets.make_blobs(n_samples=self.num_points, n_features=2, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[2] = list(datasets.make_blobs(n_samples=self.num_points, n_features=2, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[2] = list(datasets.make_blobs(n_samples=self.num_points, n_features=2, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[2] = list(datasets.make_blobs(n_samples=self.num_points, n_features=2, centers = arr, cluster_std=self.std))

		###3 clusters is equilateral triangle
		arr = np.zeros((3 ,2))
		arr[0,0:2] = [0, np.sqrt(3)*self.distance/4]
		arr[1,0:2] = [-self.distance/2, -np.sqrt(3)*self.distance/4]
		arr[2,0:2] = [self.distance/2, -np.sqrt(3)*self.distance/4]

		self.balanced_clusters_normal[3] = list(datasets.make_blobs(n_samples=self.num_points, n_features=3, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[3] = list(datasets.make_blobs(n_samples=self.num_points, n_features=3, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[3] = list(datasets.make_blobs(n_samples=self.num_points, n_features=3, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[3] = list(datasets.make_blobs(n_samples=self.num_points, n_features=3, centers = arr, cluster_std=self.std))

		###4 clusters is square
		arr = np.zeros((4 ,2))
		arr[0,0:2] = [-self.distance/2, -self.distance/2]
		arr[1,0:2] = [-self.distance/2, self.distance/2]
		arr[2,0:2] = [self.distance/2, -self.distance/2]
		arr[3,0:2] = [self.distance/2, self.distance/2]

		self.balanced_clusters_normal[4] = list(datasets.make_blobs(n_samples=self.num_points, n_features=4, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[4] = list(datasets.make_blobs(n_samples=self.num_points, n_features=4, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[4] = list(datasets.make_blobs(n_samples=self.num_points, n_features=4, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[4] = list(datasets.make_blobs(n_samples=self.num_points, n_features=4, centers = arr, cluster_std=self.std))

		###5 clusters is pentagon
		arr = np.zeros((5 ,2))
		arr[0,0:2] = [0, self.distance]
		arr[1,0:2] = [self.distance * np.cos(np.pi/10), self.distance * np.sin(np.pi/10)]
		arr[2,0:2] = [self.distance * np.cos(-np.pi * 0.3), self.distance * np.sin(-np.pi * 0.3)]
		arr[3,0:2] = [-self.distance * np.cos(np.pi/10), self.distance * np.sin(np.pi/10)]
		arr[4,0:2] = [-self.distance * np.cos(-np.pi * 0.3), self.distance * np.sin(-np.pi * 0.3)]

		self.balanced_clusters_normal[5] = list(datasets.make_blobs(n_samples=self.num_points, n_features=5, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[5] = list(datasets.make_blobs(n_samples=self.num_points, n_features=5, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[5] = list(datasets.make_blobs(n_samples=self.num_points, n_features=5, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[5] = list(datasets.make_blobs(n_samples=self.num_points, n_features=5, centers = arr, cluster_std=self.std))

		###6 clusters is hexagon
		arr = np.zeros((6 ,2))
		arr[0,0:2] = [0, self.distance]
		arr[1,0:2] = [self.distance * np.sqrt(3)/2, self.distance * 0.5]
		arr[2,0:2] = [self.distance * np.sqrt(3)/2, -self.distance * 0.5]
		arr[3,0:2] = [0, -self.distance]
		arr[4,0:2] = [-self.distance * np.sqrt(3)/2, self.distance * 0.5]
		arr[5,0:2] = [-self.distance * np.sqrt(3)/2, -self.distance * 0.5]

		self.balanced_clusters_normal[6] = list(datasets.make_blobs(n_samples=self.num_points, n_features=6, centers = arr, cluster_std=self.std))
		self.balanced_clusters_tight[6] = list(datasets.make_blobs(n_samples=self.num_points, n_features=6, centers = arr, cluster_std=self.std/2))
		self.balanced_clusters_loose[6] = list(datasets.make_blobs(n_samples=self.num_points, n_features=6, centers = arr, cluster_std=self.std*2))
		self.balanced_clusters_doubled[6] = list(datasets.make_blobs(n_samples=self.num_points, n_features=6, centers = arr, cluster_std=self.std))

		###imbalanced clusters


		###outlier clusters
		#TODO IDK HOW SHUD ADD OUTLIERS

		###simplex


		###polyhedron


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

	plt.show()