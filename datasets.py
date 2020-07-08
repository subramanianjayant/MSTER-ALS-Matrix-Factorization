import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import copy
from optimize.functions import MSTER, LCR, loss, grad, H, pairwise_square_distance, normalise
from matplotlib import pyplot as plt
from termcolor import colored

class datasets():
	def __init__(self, distance = 5, default_ratio = 10, seed = 1600, n = 2000, imbalance_ratio = 10):
		from sklearn import datasets

		self.distance = distance
		self.std_ratio = default_ratio
		self.std = distance / self.std_ratio
		self.seed = seed
		np.random.seed(self.seed)
		self.num_points = n
		self.imbalance_ratio = imbalance_ratio

		###imbalanced clusters; ONE pair of adjacent clusters is imbalanced with one having 4x the number of the other. All other clusters are unaffected
		self.imbalanced_point_distributions = {}
		for i in range(2,100):
			distribution = []
			distribution.append(int(self.imbalance_ratio * 2 * self.num_points / ((self.imbalance_ratio+1) * i)))
			distribution.append(int(2 * self.num_points / ((self.imbalance_ratio+1) * i)))
			for j in range(2, i):
				distribution.append(int(self.num_points / i))
			self.imbalanced_point_distributions[i] = distribution

		###characteristics of modified datasets
		self.characteristics = {
			"normal" : (lambda arr,k, : list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))),
			"tight" : (lambda arr,k : list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std/2))),
			"loose" : (lambda arr,k : list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std*2))),
			"double" : (lambda arr,k, : list(datasets.make_blobs(n_samples=self.num_points, n_features=k, centers = arr, cluster_std=self.std))),
			"imbalanced" : (lambda arr,k : list(datasets.make_blobs(n_samples=self.imbalanced_point_distributions[k], n_features=k, centers = arr, cluster_std=self.std)))
		}

		###dictionary format; dict[characteristic][k] returns list [dataset, labels] where k is number of clusters
		base = {}
		for key in self.characteristics.keys():
			base[key] = {}

		self._2d_clusters = copy.deepcopy(base)
		self.simplexes = copy.deepcopy(base)
		self.polyhedra = copy.deepcopy(base)

### 2D Clusters ###

		###2 clusters is just separated along x axis
		k = 2
		arr = np.zeros((k ,2))
		arr[0,0:2] = [self.distance/2, 0]
		arr[1,0:2] = [-self.distance/2, 0]

		for characteristic, function in self.characteristics.items():
			self._2d_clusters[characteristic][k] = function(arr,k)

		###3 clusters is equilateral triangle
		k=3
		arr = np.zeros((k ,2))
		arr[0,0:2] = [0, np.sqrt(3)*self.distance/4]
		arr[1,0:2] = [-self.distance/2, -np.sqrt(3)*self.distance/4]
		arr[2,0:2] = [self.distance/2, -np.sqrt(3)*self.distance/4]

		for characteristic, function in self.characteristics.items():
			self._2d_clusters[characteristic][k] = function(arr,k)

		###4 clusters is square
		k=4
		arr = np.zeros((k ,2))
		arr[0,0:2] = [-self.distance/2, -self.distance/2]
		arr[1,0:2] = [-self.distance/2, self.distance/2]
		arr[2,0:2] = [self.distance/2, -self.distance/2]
		arr[3,0:2] = [self.distance/2, self.distance/2]

		for characteristic, function in self.characteristics.items():
			self._2d_clusters[characteristic][k] = function(arr,k)

		###5 clusters is pentagon
		k=5
		arr = np.zeros((k ,2))
		l = self.distance / (2*np.sin(np.pi / 5))
		arr[0,0:2] = [0, l]
		arr[1,0:2] = [l * np.cos(np.pi * 0.1), l * np.sin(np.pi * 0.1)]
		arr[2,0:2] = [l * np.cos(-np.pi * 0.3), l * np.sin(-np.pi * 0.3)]
		arr[3,0:2] = [-l * np.cos(np.pi * 0.1), l * np.sin(np.pi * 0.1)]
		arr[4,0:2] = [-l* np.cos(-np.pi * 0.3), l * np.sin(-np.pi * 0.3)]

		print(normalise(arr))
		print(normalise(3 * arr))


		for characteristic, function in self.characteristics.items():
			self._2d_clusters[characteristic][k] = function(arr,k)

		###6 clusters is hexagon
		k=6
		arr = np.zeros((k ,2))
		arr[0,0:2] = [0, self.distance]
		arr[1,0:2] = [self.distance * np.sqrt(3)/2, self.distance * 0.5]
		arr[2,0:2] = [self.distance * np.sqrt(3)/2, -self.distance * 0.5]
		arr[3,0:2] = [0, -self.distance]
		arr[4,0:2] = [-self.distance * np.sqrt(3)/2, self.distance * 0.5]
		arr[5,0:2] = [-self.distance * np.sqrt(3)/2, -self.distance * 0.5]

		for characteristic, function in self.characteristics.items():
			self._2d_clusters[characteristic][k] = function(arr,k)

### Simplex ###
		self.max_dimension = 10
		D = self.distance**2
		for characteristic, function in self.characteristics.items():
			for d in range(2, self.max_dimension + 1):
				arr = np.zeros((d+1,d))
				arr[0] = np.full(shape = d, fill_value = -0.5)
				for i in range(1, d+1):
					for j in range(d):
						if i == j+1:
							arr[i,j] = ((D - 1) * np.sqrt(D+ 1) + 1) / (2 * D)
						else:
							arr[i,j] = (1 - np.sqrt(D + 1)) / (2 * D)
				self.simplexes[characteristic][d+1] = function(arr, d+1)
		#TODO ensure the intercluster distances between two points in the simplex is constant, independent of k

### Polyhedron ###

		self.max_polyhedron = 6

### Normalisation ###
		for characteristic in self._2d_clusters.keys():
			for k in self._2d_clusters[characteristic].keys():
				if characteristic == "double":
					self._2d_clusters[characteristic][k][0] = 200 * np.mat(np.mat(self._2d_clusters[characteristic][k][0])) / (np.trace(np.mat(self._2d_clusters[characteristic][k][0]).T * np.mat(self._2d_clusters[characteristic][k][0])))
				else:
					self._2d_clusters[characteristic][k][0] = 100 * np.mat(np.mat(self._2d_clusters[characteristic][k][0])) / (np.trace(np.mat(self._2d_clusters[characteristic][k][0]).T * np.mat(self._2d_clusters[characteristic][k][0])))
		for characteristic in self.simplexes.keys():
			for k in self.simplexes[characteristic].keys():
				if characteristic == "double":
					self.simplexes[characteristic][k][0] = 200 * np.mat(np.mat(self.simplexes[characteristic][k][0])) / (np.trace(np.mat(self.simplexes[characteristic][k][0]).T * np.mat(self.simplexes[characteristic][k][0])))
				else:
					self.simplexes[characteristic][k][0] = 100 * np.mat(np.mat(self.simplexes[characteristic][k][0])) / (np.trace(np.mat(self.simplexes[characteristic][k][0]).T * np.mat(self.simplexes[characteristic][k][0])))

	def get_datasets(self):
		return (
			self._2d_clusters,
			self.simplexes,
			self.polyhedra
		)


if __name__ == "__main__":

	_2d_clusters, simplexes, polyhedra = datasets(imbalance_ratio=10).get_datasets()
	for k in range(2,7):
		print(colored("{} clusters".format(k), "green"))
		print("Normal: {}".format(LCR(_2d_clusters["normal"][k][0],k)[0]))
	# 	print("Tight: {}".format(LCR(_2d_clusters["tight"][k][0],k)[0]))
	# 	print("loose: {}".format(LCR(_2d_clusters["loose"][k][0],k)[0]))
	# 	print("Double: {}".format(LCR(_2d_clusters["double"][k][0],k)[0]))
	# 	print("Imbalanced: {}\n\n".format(LCR(_2d_clusters["imbalanced"][k][0],k)[0]))

		# print("Normal: {}".format(LCR(simplexes["normal"][k][0],k)[0]))
	# 	print("Tight: {}".format(LCR(_2d_clusters["tight"][k][0],k)[0]))
	# 	print("loose: {}".format(LCR(_2d_clusters["loose"][k][0],k)[0]))
	# 	print("Double: {}".format(LCR(_2d_clusters["double"][k][0],k)[0]))
	# 	print("Imbalanced: {}\n\n".format(LCR(_2d_clusters["imbalanced"][k][0],k)[0]))

	# k = 6

	# plt.figure(1)
	# plt.scatter(x = np.array(_2d_clusters["normal"][k][0][:,0]), y = np.array(_2d_clusters["normal"][k][0][:,1]))
	# plt.axis('equal')
	
	# plt.figure(2)
	# plt.scatter(x = np.array(_2d_clusters["tight"][k][0][:,0]), y = np.array(_2d_clusters["tight"][k][0][:,1]))
	# plt.axis('equal')
	
	# plt.figure(3)
	# plt.scatter(x = np.array(_2d_clusters["loose"][k][0][:,0]), y = np.array(_2d_clusters["loose"][k][0][:,1]))
	# plt.axis('equal')
	
	# plt.figure(4)
	# plt.scatter(x = np.array(_2d_clusters["double"][k][0][:,0]), y = np.array(_2d_clusters["double"][k][0][:,1]))
	# plt.axis('equal')
	
	# plt.figure(5)
	# plt.scatter(x = np.array(_2d_clusters["imbalanced"][k][0][:,0]), y = np.array(_2d_clusters["imbalanced"][k][0][:,1]))
	# plt.axis('equal')

	# from mpl_toolkits.mplot3d import Axes3D
	# fig = plt.figure(6)
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(simplexes["imbalanced"][4][0][:,0], simplexes["imbalanced"][4][0][:,1], simplexes["imbalanced"][4][0][:,2])

	plt.show()
