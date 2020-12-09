import numpy as np
from sklearn import metrics
import scipy
import math
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.special import comb
import itertools

def get_dendrogram_weights(x, method = 'ward'):
    ytdist = distance.pdist(x, metric = 'euclidean')
    Z = hierarchy.linkage(ytdist, method)
    dn = hierarchy.dendrogram(Z)

    icoord = scipy.array(dn['icoord'])
    dcoord = scipy.array(dn['dcoord'])
    x = sorted(list(zip(dn['dcoord'])), key = lambda x: x[0][1], reverse = True)
#    print((x))
    height = x[0][0][1]
    weights = []
    x.append(([0,0,0,0],))
    while len(x) > 1:
#        print(x[0])
        weights.append(x[0][0][1] - x[1][0][1])
        x.pop(0)
#        print(len(x))
    weights = weights / height
    return weights