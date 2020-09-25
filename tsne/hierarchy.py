import numpy as np
from sklearn import metrics
import scipy
import math

def merge(arr, delta, cut_D=[]):
    n = len(arr)
    if len(cut_D) == 0:
        D = np.array([[delta(i,j) for i in arr] for j in arr])  # -> OPTIMIZED THIS WITH MEMOIZATION
    else:
        newrow = np.array([delta(arr[-1],i) for i in arr])
        D = np.zeros((n,n))
        D[0:len(cut_D),0:len(cut_D)] = cut_D.copy()
        D[-1] = newrow
        D[:,-1] = newrow

    D[range(n),range(n)] = np.inf
    a = np.argwhere(D == np.min(D))[0]
    min_, max_ = min(a), max(a)
    cut_D = np.delete(np.delete(D, [min_,max_],0), [min_,max_], 1)
    return arr[0:min_]+arr[min_+1:max_]+arr[max_+1:n]+[arr[min_]+arr[max_]], D, cut_D

def calc_partitions(arr, delta): # -> SWITCH TO KEEPING TRACK OF INDICES INSTEAD OF POINTS SO CORRECT P/Q POINTS ALIGNED ACROSS PARTITIONS
    partitions = []
    pdists = []
    X = [[r] for r in arr]
    cut_D = []
    while len(X)>1:
        n = len(X)
        partitions.append(X.copy())
        X, D, cut_D = merge(X, delta, cut_D)
        D[range(n),range(n)] = 0.
        pdists.append(D)
    return partitions, pdists

def dist(x,y):
    return np.linalg.norm(np.mean(y, axis=0)-np.mean(x, axis=0))**2

if __name__ == '__main__':
    parts,dists = calc_partitions([[3, 1], [4, 4], [4, 6], [2, 2], [2, 2]], dist)
    for part in parts:
        print(part)
