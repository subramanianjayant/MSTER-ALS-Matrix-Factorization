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

<<<<<<< HEAD
desired_classes = [0,1,8]
num_points = 200

df = pd.read_csv('mnist_784_zip/data/mnist_784_csv.csv')
df = df.loc[df['class'].isin(desired_classes)]
df = df.sample(n=num_points, random_state=1600)
labels = np.array(df['class'])
data = np.mat(df.drop('class', axis=1))

pca_0 = PCA(n_components = 30) #initial dim reduction for faster MST computation (from tSNE paper)
init_data = np.mat(pca_0.fit_transform(data))
(M, A, B, k_A, lr_a, lr_a_decay, lr_b, lr_b_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs, clip_a, clip_b, random_state) = MFConfig(M=init_data).dump()
assert len(desired_classes) == k_A
=======
num_points = 200
num_clusters = 5
random_state = 1600

################## SAMPLE DATASET ################
from sklearn import datasets

arr = np.zeros((5,20))
arr[0,0:3] = [5, 5,-10]
arr[1,0:3] = [5,-5,-10]
arr[2,0:3] = [-5,5,-10]
arr[3,0:3] = [-5,-5,-10]
arr[4,2] = 20
init_data, labels = datasets.make_blobs(n_samples=num_points, n_features=20, centers = arr, cluster_std=0.2)
init_data = np.mat(init_data)
synth_pca = PCA(n_components=2)
pca_init = synth_pca.fit_transform(init_data)
(M, A, B, k_A, lr_a, lr_a_decay, lr_b, lr_b_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs, clip_a, clip_b) = MFConfig(M=init_data).dump()
##################################################

# df = pd.read_csv('mnist_784_zip/data/mnist_784_csv.csv')
# df = df.loc[df['class'].isin([0,1,4,6,8])]
# df = df.sample(n=num_points, random_state=random_state)
# labels = np.array(df['class'])
# data = np.mat(df.drop('class', axis=1))

# pca_0 = PCA(n_components = 30) #initial dim reduction for faster MST computation (from tSNE paper)
# init_data = np.mat(pca_0.fit_transform(data))
# (M, A, B, k_A, lr_a, lr_a_decay, lr_b, lr_b_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs, clip_a, clip_b) = MFConfig(M=init_data).dump()
>>>>>>> 25c045ce8f4d8f74248c56d105c71585b9f53948

# pca = PCA(n_components = 2)
# pca_init = np.mat(pca.fit_transform(init_data))

def train():
    global M, A, B, k_A, lr_a, lr_a_decay, lr_b, lr_b_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs, clip_a, clip_b
    A_best = A.copy()
    B_best = B.copy()

    # ratio_A, vertices_A = MSTER(A, k_A)
    # ratio_B, vertices_B = MSTER(B.T, k_B)
    ratio_A, A_A, B_A, C_A, D_A = LCR(A, k_A)
    loss_best = loss(M,A,B,ratio_A, lambda_, eta)
    for epoch in range(num_epochs):
        best = ''
        if (epoch%10)<5:
            # gradient = (-(M-A*B)*B.T - lambda_*grad(A,vertices_A) - eta*(2*H(M.shape[0]).T*H(M.shape[0])*A))
            gradient = (-(M-A*B)*B.T - lambda_*grad(A,A_A, B_A, C_A, D_A) - eta*(2*H(M.shape[0]).T*H(M.shape[0])*A))
            n = np.linalg.norm(gradient, ord='fro')
            if n > clip_a:
                gradient = clip_a * gradient/n

            A = A - lr_a*gradient
        else:
            gradient = (-A.T*(M-A*B))
            n = np.linalg.norm(gradient, ord='fro')
            if n > clip_b:
                gradient = clip_b * gradient/n

            B = B - lr_b*gradient

        # ratio_A, vertices_A = MSTER(A, k_A)
        # ratio_B, vertices_B = MSTER(B.T, k_B)
        ratio_A, A_A, B_A, C_A, D_A = LCR(A, k_A)
        loss_ = loss(M,A,B,ratio_A, lambda_, eta)

        if loss_<loss_best:
            A_best = A.copy()
            B_best = B.copy()
            best = 'best'
            loss_best = loss_

        lr_a -= lr_a_decay
        lr_b -= lr_b_decay
        lambda_ -= lambda_decay
        eta -= eta_decay

        print("epoch {0} --- \t loss: {1} \t norm contribution: {2} \t {3}".format(epoch,
                loss_, 0.5*np.linalg.norm(M-A*B, ord = 'fro')**2, best))

    return A_best,B_best #returns best model in terms of loss

if __name__ == '__main__':
    A_best, B_best = train()

    ### Dim Reduction
<<<<<<< HEAD
    base = {} 
    for class_ in desired_classes:
        base[class_] = []
    base_kmeans = {}
    for i in range(len(desired_classes)):
        base_kmeans[i] = []
=======
    #base = {1: [], 4: [], 6: [], 8: [], 0: []} #{1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    base = {0: [], 1: [], 2: [], 3: [], 4: []}
    base_kmeans = {0: [], 1: [], 2: [], 3:[], 4:[]}
    #legend = [1,4,6,8,0]
    legend = [0,1,2,3,4]
>>>>>>> 25c045ce8f4d8f74248c56d105c71585b9f53948

    _dict = copy.deepcopy(base)
    for i,row in enumerate(np.array(A_best)):
        _dict[labels[i]].append(row)

    plt.figure(1)
    plt.title('d=2 MSTER-ALS Representation of MNIST Sample (n={})'.format(num_points))
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

    predictions_MSTER = KMeans(n_clusters = k_A, random_state = random_state).fit(A_best).labels_
    predictions_PCA = KMeans(n_clusters = k_A, random_state = random_state).fit(pca_init).labels_

    plt.figure(3)
    plt.title('K-Means predictions for MSTER')
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

    print("MSTER NMI score: {} \n PCA NMI score: {}".format(score_MSTER, score_PCA))

    ### ARI
    score_MSTER = adjusted_rand_score(labels, predictions_MSTER)
    score_PCA = adjusted_rand_score(labels, predictions_PCA)

    print("MSTER Rand score: {} \n PCA Rand score: {}".format(score_MSTER, score_PCA))

    print("lambda={}_rand={}".format(lambda_, random_state))

    plt.show()

    
