import numpy as np
from sklearn import metrics
import scipy
import math
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance

def merge(arr, delta, idxs, prev_D=[], cut_D=[]):
    idxs_ = idxs.copy()
    if len(cut_D) == 0:
        D = np.array([[delta(i,j) for i in arr] for j in arr])  # -> OPTIMIZED THIS WITH MEMOIZATION
        full_D = D.copy()
        n = len(arr)
        N = len(arr)
    else:
        N = len(prev_D)
        newrow_ = np.zeros(N)
        for i in range(len(idxs)):
            d = delta(arr[-1],arr[i])
            for j in idxs[i]:
                newrow_[j] = d
        newidxs = idxs[-1]
        for i in newidxs:
            newrow_[i] = np.inf
        full_D = prev_D.copy()
        for i in newidxs:
            full_D[i] = newrow_
            full_D[:,i] = newrow_

        n=len(arr)
        newrow = np.array([delta(arr[-1], i) for i in arr])
        D = np.zeros((n,n))
        D[0:len(cut_D),0:len(cut_D)] = cut_D.copy()
        D[-1] = newrow
        D[:,-1] = newrow

    D[range(n),range(n)] = np.inf
    full_D[range(N),range(N)] = np.inf
    a = np.argwhere(D == np.min(D))[0]
    min_, max_ = min(a), max(a)
    cut_D = np.delete(np.delete(D, [min_,max_],0), [min_,max_], 1)
    for i in idxs[-1]:
        full_D[[i],idxs[-1]] = 0.
    idxs_ = idxs_[0:min_]+idxs_[min_+1:max_]+idxs_[max_+1:n]+[idxs_[min_]+idxs_[max_]]
    return arr[0:min_]+arr[min_+1:max_]+arr[max_+1:n]+[arr[min_]+arr[max_]], idxs_, D, cut_D, full_D

def calc_partitions(arr, delta): # -> SWITCH TO KEEPING TRACK OF INDICES INSTEAD OF POINTS SO CORRECT P/Q POINTS ALIGNED ACROSS PARTITIONS
    partitions = []
    pdists = []
    X = [[r] for r in arr]
    idxs = [[i] for i in range(len(arr))]
    cut_D = []
    full_D = []
    n = len(X)
    while len(X)>1:
        #partitions.append(X.copy())
        partitions.append(idxs.copy())
        X, idxs, D, cut_D, full_D = merge(X, delta, idxs, full_D, cut_D)
        full_D[range(n),range(n)] = 0.
        pdists.append(full_D)
    return partitions, pdists

def calc_dists(X, partitions, delta):
    dist_arr = []
    (n,d) = X.shape
    for p in partitions:
        if len(p) == n:
            D_ = np.array([[delta([i],[j]) for i in X] for j in X])
        else:
            newrow_ = np.zeros(n)
            for i in range(len(p)):
                d = delta(X[p[-1]],X[p[i]])
                for j in p[i]:
                    newrow_[j] = d
            newidxs = p[-1]
            for i in newidxs:
                D_[i] = newrow_
                D_[:,i] = newrow_
        dist_arr.append(D_.copy())
    return dist_arr

def dist(x,y):
    return (len(x)*len(y))*np.linalg.norm(np.mean(y, axis=0)-np.mean(x, axis=0))**2/(len(x)+len(y))

def weighted_nmi(original_X, new_X):
    weights = get_dendrogram_weights(original_X)[::-1]
    original_partitions, _ = calc_partitions(original_X, dist)
    new_partitions, _ = calc_partitions(new_X, dist)

    nmi = []
    for p, pp in zip(original_partitions, new_partitions):
        old_labels = []
        new_labels = []
        label_counter = 0
        for g, gg in zip(p,pp):
            old_labels.extend([label_counter]*len(g))
            new_labels.extend([label_counter]*len(gg))
            label_counter += 1
        nmi.append(metrics.normalized_mutual_info_score(old_labels, new_labels))
    return np.dot(np.array(nmi), np.array(weights))

def fowlkes_mallows_indices(original_X, new_X):
    weights = get_dendrogram_weights(original_X)[::-1]
    original_partitions, _ = calc_partitions(original_X, dist)
    new_partitions, _ = calc_partitions(new_X, dist)
    n = len(original_partitions)

    indices = {}
    for p, pp in zip(original_partitions, new_partitions):
        k = len(p)
        old_labels = []
        new_labels = []
        label_counter = 0
        for g, gg in zip(p,pp):
            old_labels.extend([label_counter]*len(g))
            new_labels.extend([label_counter]*len(gg))
            label_counter += 1
        indices[k] = np.sqrt(metrics.precision_score(old_labels, new_labels, average='micro')*metrics.recall_score(old_labels, new_labels, average='micro'))
    plt.figure()
    plt.plot(indices.keys(), indices.values())
    plt.show()
    return indices

#takes in high-dimensional data, outputs lengths of branch segments
#Outputs weights as % of total height, ordered from first cluster merge to last cluster merge
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

if __name__ == '__main__':
    data = np.array([[3, 1], [4, 4], [4, 6], [2, 2], [2, 2]])
    parts,dists = calc_partitions(data, dist)
    for part in parts:
        print(part)
    print(fowlkes_mallows_indices(data, data))
