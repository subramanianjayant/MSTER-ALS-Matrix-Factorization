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

DATASET = 'MNIST' #supports IRIS, WINE, MNIST, RADAR, PROTEIN, OCTAHEDRON
desired_classes = [0,1,6,9]
# desired_classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
num_points = 200
random_state = 1600
d = 2
np.random.seed(random_state)

num_clusters = len(desired_classes)
epoch = 1

(M, P, k, lambda_, lr, num_epochs, random_state, method, pca_init, labels) = MFConfig(dataset=DATASET, desired_classes=desired_classes, k=num_clusters, num_points=num_points, seed=random_state, d=d).dump()

def train():
    #assert num_clusters == len(desired_classes)
    global M, P, k, lambda_, lr, num_epochs, random_state, method

    M = normalise(M)

    if method.lower() == 'adam':
        P_best, loss_best = adam((P, M, k, lambda_), num_epochs = num_epochs, alpha = lr)
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

        # plt.figure(1)
        # plt.title('d=2 LCR Representation of {1} (n={0}, RAND={2}, NMI={3})'.format(num_points, DATASET, round(rand_score_MSTER,2), round(nmi_score_MSTER, 2)))
        # # print(_dict)
        # for i in _dict.keys():
        #     plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
        # plt.legend(desired_classes)
        # plt.savefig('figures/LCR_{1}_n={0}_classes={2}.png'.format(num_points, DATASET, str(desired_classes)))
        #
        # plt.figure(2)
        # plt.title('d=2 PCA Representation of {1} (n={0}, RAND={2}, NMI={3})'.format(num_points, DATASET, round(rand_score_PCA,2), round(nmi_score_PCA,2)))
        # for i in _dict2.keys():
        #     plt.scatter(np.array(_dict2[i])[:,0], np.array(_dict2[i])[:,1], alpha=0.6)
        # plt.legend(desired_classes)
        # plt.savefig('figures/PCA_{1}_n={0}_classes={2}.png'.format(num_points, DATASET, str(desired_classes)))

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
    # else:
    #     plt.figure(1)
    #     plt.title('d=2 LCR Representation of {1} (n={0}, k={2})'.format(num_points, DATASET, num_clusters))
    #     # print(_dict)
    #     for i in _dict.keys():
    #         plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
    #     plt.legend(desired_classes)
    #     plt.savefig('figures/PCA_{1}_n={0}_k={2})'.format(num_points, DATASET, num_clusters))
    #
    #     plt.figure(2)
    #     plt.title('d=2 PCA Representation of {1} (n={0}, k={2})'.format(num_points, DATASET, num_clusters))
    #     for i in _dict2.keys():
    #         plt.scatter(np.array(_dict2[i])[:,0], np.array(_dict2[i])[:,1], alpha=0.6)
    #     plt.legend(desired_classes)
    #     plt.savefig('figures/PCA_{1}_n={0}_k={2})'.format(num_points, DATASET, num_clusters))
    #
    #
    # plt.show()
