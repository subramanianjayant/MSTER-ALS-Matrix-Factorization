import numpy as np

def merge(A, B, cost):

	size_a = len(A)
	size_b = len(B)

	mean_a = np.ones(A.shape[0]) * A / size_a
	mean_b = np.ones(B.shape[0]) * B / size_b


	dist = np.sqrt((size_a + size_b) * cost / (size_a * size_b))

	trans = (mean_b - mean_a + np.array([[dist, 0]]))

	B = B-trans

	return np.vstack([A,B])

if __name__ == "__main__":
	print(merge(np.mat([[1,1],[2,2]]),np.mat([[50,50],[51,51]]), 200))

