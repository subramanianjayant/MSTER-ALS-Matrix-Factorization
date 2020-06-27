import numpy as np
from sklearn.metrics import pairwise, normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
from matplotlib import pyplot as plt
from optimize.functions import MSTER, LCR, loss, grad, H
from sklearn.cluster import KMeans
from config import MFConfig
import pandas as pd
from sklearn.decomposition import PCA
import copy
from termcolor import colored

num_points = 200
random_state = 1600
np.random.seed(random_state)

################## SAMPLE DATASET (OCTAHEDRON )################
# from sklearn import datasets
#
# num_dimensions = 20
# num_clusters = 4
#
# arr = np.zeros((6 ,num_dimensions))
# arr[0,0:3] = [5, 5, 0]
# arr[1,0:3] = [5, -5, 0]
# arr[2,0:3] = [-5, 5, 0]
# arr[3,0:3] = [-5,-5, 0]
# arr[4,0:3] = [0, 0, 5]
# arr[5,0:3] = [0, 0, -5]
# init_data, labels = datasets.make_blobs(n_samples=num_points, n_features=num_dimensions, centers = arr, cluster_std=0.6)
# init_data = np.mat(init_data)
# synth_pca = PCA(n_components=2)
# pca_init = synth_pca.fit_transform(init_data)
# (M, P, k, lr, lr_decay, num_epochs, clip, random_state) = MFConfig(M=init_data, k=num_clusters).dump()
# desired_classes = [0,1,2,3,4,5]
##################################################

################### SAMPLE DATASET (GAUSSIAN CENTER DRAWS) ###########
# from sklearn import datasets
#
# num_dimensions = 10
# variance_ratio = 10000
#
# arr = np.zeros((num_clusters ,num_dimensions))
#
# for x in range(num_clusters):
#     arr[x] = np.random.multivariate_normal(np.zeros(num_dimensions), (variance_ratio**0.5)*np.identity(num_dimensions))
#
# init_data, labels = datasets.make_blobs(n_samples=num_points, n_features=num_dimensions, centers = arr, cluster_std=1)
# init_data = np.mat(init_data)
# synth_pca = PCA(n_components=2)
# pca_init = synth_pca.fit_transform(init_data)
# (M, P, k, lr, lr_decay, num_epochs, clip, random_state) = MFConfig(M=init_data, k=num_clusters, seed = random_state).dump()

#####################################################################

################## MNIST ###########################
desired_classes = [0,1,8]
num_clusters = len(desired_classes)

df = pd.read_csv('mnist_784_zip/data/mnist_784_csv.csv')
df = df.loc[df['class'].isin(desired_classes)]
df = df.sample(n=num_points, random_state=random_state)
labels = np.array(df['class'])
data = np.mat(df.drop('class', axis=1))

pca_0 = PCA(n_components = 60) #initial dim reduction for faster MST computation (from tSNE paper)
init_data = np.mat(pca_0.fit_transform(data))
(M, P, k, lr, lr_decay, num_epochs, clip, random_state) = MFConfig(M=init_data, k=num_clusters, seed = random_state).dump()
synth_pca = PCA(n_components = 2)
pca_init = np.mat(synth_pca.fit_transform(init_data))
#####################################################

def train():
    # assert num_clusters == len(desired_classes)
    global M, P, k, lr, lr_decay, num_epochs, clip
    P_best = P.copy()

    # ratio, vertices = MSTER(M*P, k)
    ratio, A, B, C, D = LCR(M*P, k)
    loss_best = loss(ratio)
    for epoch in range(num_epochs):
        best = ''
        gradient = grad(M, P, A, B, C, D)
        n = np.linalg.norm(gradient, ord='fro')
        if n > clip:
            gradient = clip * gradient/n

        P = P + lr*gradient

        # ratio, vertices = MSTER(M*P, k)
        ratio, A, B, C, D = LCR(M*P, k)
        loss_ = loss(ratio)

        if loss_>loss_best:
            P_best = P.copy()
            best = 'best'
            loss_best = loss_

        lr -= lr_decay

        print("epoch {0} --- \t loss: {1} \t {2}".format(epoch, loss_, best))

    return P, loss_best #returns latest model
    #return P_best #returns best model in terms of loss

if __name__ == '__main__':
    P_best, loss_best = train()
    A_best = M*P_best
    ### Dim Reduction

    ############## SAMPLE DATASETS ###############
    # base = {0: [], 1: [], 2: [], 3: [], 4: [], 5:[]}
    # base_kmeans = {0: [], 1: [], 2: [], 3: [], 4: [], 5:[]}
    # legend = [0,1,2,3,4,5]
    ################################################

    ############### MNIST #########################
    base = {}
    for class_ in desired_classes:
        base[class_] = []
    base_kmeans = {}
    for i in range(len(desired_classes)):
        base_kmeans[i] = []
    ################################################

    _dict = copy.deepcopy(base)
    for i,row in enumerate(np.array(A_best)):
        _dict[labels[i]].append(row)

    plt.figure(1)
    plt.title('d=2 LCR-ALS Representation of MNIST Sample (n={})'.format(num_points))
    # print(_dict)
    for i in _dict.keys():
        plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
    plt.legend(desired_classes)

    _dict2 = copy.deepcopy(base)
    for i,row in enumerate(np.array(pca_init)):
        _dict2[labels[i]].append(row)

    plt.figure(2)
    plt.title('d=2 PCA Representation of MNIST Sample (n={})'.format(num_points))
    for i in _dict2.keys():
        plt.scatter(np.array(_dict2[i])[:,0], np.array(_dict2[i])[:,1], alpha=0.6)
    plt.legend(desired_classes)

    ### KMeans stuff

    predictions_MSTER = KMeans(n_clusters = k, random_state = random_state).fit(A_best).labels_
    predictions_PCA = KMeans(n_clusters = k, random_state = random_state).fit(pca_init).labels_

    plt.figure(3)
    plt.title('K-Means predictions for LCR')
    _dict3 = copy.deepcopy(base_kmeans)
    for i,row in enumerate(np.array(A_best)):
        _dict3[predictions_MSTER[i]].append(row)
    for i in _dict3.keys():
        plt.scatter(np.array(_dict3[i])[:,0], np.array(_dict3[i])[:,1], alpha=0.6)
    plt.legend(range(10))

    plt.figure(4)
    plt.title('K-Means predictions for PCA')
    _dict4 = copy.deepcopy(base_kmeans)
    for i,row in enumerate(np.array(pca_init)):
        _dict4[predictions_PCA[i]].append(row)
    for i in _dict4.keys():
        plt.scatter(np.array(_dict4[i])[:,0], np.array(_dict4[i])[:,1], alpha=0.6)
    plt.legend(range(10))

    ### NMI
    score_MSTER = normalized_mutual_info_score(labels, predictions_MSTER)
    score_PCA = normalized_mutual_info_score(labels, predictions_PCA)

    print(colored("LCR NMI score: {} \n PCA NMI score: {}".format(score_MSTER, score_PCA),"green"))

    ### ARI
    score_MSTER = adjusted_rand_score(labels, predictions_MSTER)
    score_PCA = adjusted_rand_score(labels, predictions_PCA)

    # pca_score, A, B, C, D = LCR(pca_init, k)
    # print(colored("LCR norm score: {} \t PCA norm score: {}".format(loss_best, pca_score), "red"))

    print(colored("LCR Rand score: {} \n PCA Rand score: {}".format(score_MSTER, score_PCA), "green"))
    print(colored("lr={} \t rand_state={}".format(lr, random_state), "blue"))
    plt.show()
