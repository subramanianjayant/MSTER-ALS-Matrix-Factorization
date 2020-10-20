import scipy
from scipy.cluster import hierarchy
from scipy.spatial import distance
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

fname = 'tsne/htsne_data.npy'
hd_fname = 'tsne/htsne_data_hd.npy'
label_fname = 'tsne/htsne_labels.npy'
method = 'ward'

for filename in [fname, hd_fname]:
    data = np.mat(np.load(filename))
    ytdist = distance.pdist(data, metric = 'euclidean')
    Z = hierarchy.linkage(ytdist, method)
    
    plt.figure()
    labels_ = np.load(label_fname)
    dn = hierarchy.dendrogram(Z, labels = labels_)
    plt.show()
    #plt.savefig("dendrogram" + filename[5:] + ".png", dpi = 600)
