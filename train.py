import numpy as np
from sklearn.metrics import pairwise, normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
from matplotlib import pyplot as plt
from optimize.functions import LCR, loss, grad, H, normalise, calc_objective, adam, VGA, nn_score
from sklearn.cluster import KMeans
from config import MFConfig
import pandas as pd
from sklearn.decomposition import PCA
import copy
from termcolor import colored
from scipy import optimize

num_points = 200
random_state = 1600
np.random.seed(random_state)
epoch = 1

################## SAMPLE DATASET (OCTAHEDRON )################
# from sklearn import datasets
#
# DATASET = 'octahedron'
# num_dimensions = 20
# desired_classes = [0,1,2,3,4,5]
# num_clusters = len(desired_classes)
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
# (M, P, k, lambda_, lr, num_epochs, random_state, method) = MFConfig(M=init_data, k=num_clusters, seed = random_state).dump()
##################################################

################### SAMPLE DATASET (GAUSSIAN CENTER DRAWS) ###########
# from sklearn import datasets
#
# DATASET = 'Gaussian Center Draws'
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
# (M, P, k, lambda_, lr, num_epochs, random_state, method) = MFConfig(M=init_data, k=num_clusters, seed = random_state).dump()
#####################################################################

################## IRIS ############################
# DATASET = 'IRIS'
# desired_classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
# # desired_classes = ['Iris-versicolor','Iris-virginica']
# num_clusters = len(desired_classes)
#
# df = pd.read_csv('iris.data', header = None)
# df = df.loc[df[4].isin(desired_classes)]
# if num_points < len(df):
#     df = df.sample(n=num_points, random_state=random_state)
# labels = np.array(df[4])
# data = np.mat(df.drop(4, axis=1))
# print(labels)
# print(data)
# synth_pca = PCA(n_components=2)
# pca_init = synth_pca.fit_transform(data)
# (M, P, k, lambda_, lr, num_epochs, random_state, method) = MFConfig(M=data, k=num_clusters, seed = random_state).dump()
####################################################

################## WINE ############################
# DATASET = 'WINE'
# desired_classes = [1,2,3]
# num_clusters = len(desired_classes)
#
# df = pd.read_csv('wine.data', header = None)
# df = df.loc[df[0].isin(desired_classes)]
# if num_points < len(df):
#     df = df.sample(n=num_points, random_state=random_state)
# labels = np.array(df[0])
# data = np.mat(df.drop(0, axis=1))
# (M, P, k, lambda_, lr, num_epochs, random_state, method) = MFConfig(M=data, k=num_clusters, seed = random_state).dump()
# synth_pca = PCA(n_components = 2)
# pca_init = np.mat(synth_pca.fit_transform(data))
####################################################

################## PROTEIN #########################
# DATASET = 'PROTEIN'
# desired_classes = ['A','B','E']
# num_clusters = len(desired_classes)
#
# df = pd.read_csv('nmMDS_131d.csv', header = None)
# labels = pd.read_csv('protein_labels.csv', header = None)
# df['labels'] = np.array(labels[1])
# df = df.loc[df['labels'].isin(desired_classes)]
# if num_points < len(df):
#     df = df.sample(n=num_points, random_state=random_state)
# labels = np.array(df['labels'])
# data = np.mat(df.drop('labels', axis=1))
# (M, P, k, lambda_, lr, num_epochs, random_state, method) = MFConfig(M=data, k=num_clusters, seed = random_state).dump()
# synth_pca = PCA(n_components = 2)
# pca_init = np.mat(synth_pca.fit_transform(data))
####################################################

################## RADAR #########################
# DATASET = 'RADAR'
# desired_classes = ['B','G']
# num_clusters = len(desired_classes)

# df = pd.read_csv('RADAR_mMDS_351d.csv', header = None)
# labels = pd.read_csv('radar_labels.csv', header = None)
# df['labels'] = np.array(labels[1])
# df = df.loc[df['labels'].isin(desired_classes)]
# if num_points < len(df):
#     df = df.sample(n=num_points, random_state=random_state)
# labels = np.array(df['labels'])
# data = np.mat(df.drop('labels', axis=1))
# pca_0 = PCA(n_components = 30) #initial dim reduction for faster MST computation (from tSNE paper)
# init_data = np.mat(pca_0.fit_transform(data))
# (M, P, k, lambda_, lr, num_epochs, random_state, method) = MFConfig(M=init_data, k=num_clusters, seed = random_state).dump()
# synth_pca = PCA(n_components = 2)
# pca_init = np.mat(synth_pca.fit_transform(init_data))
####################################################

################## MNIST ###########################
DATASET = 'MNIST'
desired_classes = [0,1,8]
num_clusters = len(desired_classes)

df = pd.read_csv('mnist_784_zip/data/mnist_784_csv.csv')
df = df.loc[df['class'].isin(desired_classes)]
df = df.sample(n=num_points, random_state=random_state)
labels = np.array(df['class'])
data = np.mat(df.drop('class', axis=1))

pca_0 = PCA(n_components = 30) #initial dim reduction for faster MST computation (from tSNE paper)
init_data = np.mat(pca_0.fit_transform(data))
(M, P, k, lambda_, lr, num_epochs, random_state, method) = MFConfig(M=init_data, k=num_clusters, seed = random_state).dump()
synth_pca = PCA(n_components = 2)
pca_init = np.mat(synth_pca.fit_transform(init_data))
#####################################################

def train():
    #assert num_clusters == len(desired_classes)
    global M, P, k, lambda_, lr, num_epochs, random_state, method

    M = normalise(M)

    if method.lower() == 'adam':
        P_best = adam((P, M, k, lambda_), num_epochs = num_epochs, alpha = lr)
    elif method.lower() == 'gradient ascent':
        P_best = VGA((P, M, k, lambda_), num_epochs = num_epochs, lr = lr)
    else:
        P_arr = optimize.minimize(calc_objective, np.array(P).reshape(-1), args = (M,k,lambda_), method = method, options = {'maxiter': num_epochs}, callback = log_callback)
        P_best = np.mat(P_arr.x.reshape(P.shape))

    return P_best

def log_callback(xk):
    global epoch
    print("epoch {0} --- \t negative loss: {1}".format(epoch, calc_objective(xk, M, k, lambda_)))
    epoch +=1
    return epoch > num_epochs

if __name__ == '__main__':
    P_best = train()
    A_best = M*P_best
    np.savetxt("{}_proj_data.csv".format(DATASET), A_best, delimiter=",")

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

    _dict2 = copy.deepcopy(base)
    for i,row in enumerate(np.array(pca_init)):
        _dict2[labels[i]].append(row)

    ## KMeans stuff
    if num_clusters == len(desired_classes):
        predictions_MSTER = KMeans(n_clusters = k, random_state = random_state).fit(A_best).labels_
        predictions_PCA = KMeans(n_clusters = k, random_state = random_state).fit(pca_init).labels_

        ### NMI
        nmi_score_MSTER = normalized_mutual_info_score(labels, predictions_MSTER)
        nmi_score_PCA = normalized_mutual_info_score(labels, predictions_PCA)
        print(colored("LCR NMI score: {} \n PCA NMI score: {}".format(nmi_score_MSTER, nmi_score_PCA),"green"))

        ### ARI
        rand_score_MSTER = adjusted_rand_score(labels, predictions_MSTER)
        rand_score_PCA = adjusted_rand_score(labels, predictions_PCA)
        print(colored("LCR Rand score: {} \n PCA Rand score: {}".format(rand_score_MSTER, rand_score_PCA), "green"))
        
        ### NN purity score
        nn_score_MSTER = nn_score(A_best, labels)
        nn_score_PCA = nn_score(pca_init, labels)
        print(colored("LCR NN score: {} \n PCA NN score: {}".format(nn_score_MSTER, nn_score_PCA), "green"))

        ### Params
        print(colored("method={} \t lambda={} \t rand_state={}".format(method, lambda_, random_state), "blue"))

        ### PLOTTING

        plt.figure(1)
        plt.title('d=2 LCR Representation of {1} (n={0}, RAND={2}, NMI={3})'.format(num_points, DATASET, round(rand_score_MSTER,2), round(nmi_score_MSTER, 2)))
        # print(_dict)
        for i in _dict.keys():
            plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
        plt.legend(desired_classes)
        plt.savefig('figures/LCR_{1}_n={0}_RAND={2}_NMI={3}.png'.format(num_points, DATASET, rand_score_MSTER, nmi_score_MSTER))

        plt.figure(2)
        plt.title('d=2 PCA Representation of {1} (n={0}, RAND={2}, NMI={3})'.format(num_points, DATASET, round(rand_score_PCA,2), round(nmi_score_PCA,2)))
        for i in _dict2.keys():
            plt.scatter(np.array(_dict2[i])[:,0], np.array(_dict2[i])[:,1], alpha=0.6)
        plt.legend(desired_classes)
        plt.savefig('figures/PCA_{1}_n={0}_RAND={2}_NMI={3}.png'.format(num_points, DATASET, rand_score_PCA, nmi_score_PCA))

        # plt.figure(3)
        # plt.title('K-Means predictions for LCR')
        # _dict3 = copy.deepcopy(base_kmeans)
        # for i,row in enumerate(np.array(A_best)):
        #     _dict3[predictions_MSTER[i]].append(row)
        # for i in _dict3.keys():
        #     plt.scatter(np.array(_dict3[i])[:,0], np.array(_dict3[i])[:,1], alpha=0.6)
        # plt.legend(range(10))
        #
        # plt.figure(4)
        # plt.title('K-Means predictions for PCA')
        # _dict4 = copy.deepcopy(base_kmeans)
        # for i,row in enumerate(np.array(pca_init)):
        #     _dict4[predictions_PCA[i]].append(row)
        # for i in _dict4.keys():
        #     plt.scatter(np.array(_dict4[i])[:,0], np.array(_dict4[i])[:,1], alpha=0.6)
        # plt.legend(range(10))
    else:
        plt.figure(1)
        plt.title('d=2 LCR Representation of {1} (n={0}, k={2})'.format(num_points, DATASET, num_clusters))
        # print(_dict)
        for i in _dict.keys():
            plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
        plt.legend(desired_classes)
        plt.savefig('figures/PCA_{1}_n={0}_k={2})'.format(num_points, DATASET, num_clusters))

        plt.figure(2)
        plt.title('d=2 PCA Representation of {1} (n={0}, k={2})'.format(num_points, DATASET, num_clusters))
        for i in _dict2.keys():
            plt.scatter(np.array(_dict2[i])[:,0], np.array(_dict2[i])[:,1], alpha=0.6)
        plt.legend(desired_classes)
        plt.savefig('figures/PCA_{1}_n={0}_k={2})'.format(num_points, DATASET, num_clusters))


    plt.show()
