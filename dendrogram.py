from scipy.cluster import hierarchy
from scipy.spatial import distance
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

filename = 'radar.csv'
method = 'ward'

assert(filename!='')

data = np.mat(pd.read_csv(filename, header = None))
ytdist = distance.pdist(data, metric = 'euclidean')
Z = hierarchy.linkage(ytdist, method)

plt.figure()
dn = hierarchy.dendrogram(Z)
plt.show()
