import math
import numpy as np
import pandas as pd
from sklearn import datasets, decomposition, metrics
from matplotlib import pyplot as plt
from hierarchy import dist, merge, calc_partitions, calc_dists
import scipy
from scipy.cluster import hierarchy
from scipy.spatial import distance
import copy
import pickle
############# CONSTANTS ##############

DATA_SIZE = 300
ITERATIONS = 600

######################################


############## METHODS ###############

def perplex_helper(Di, sigma):
    Pi = np.exp(-1*Di.copy() * sigma)
    Psum = np.sum(Pi)
    perp_calc = np.log(Psum) + sigma * np.sum(Di * Pi) / Psum
    Pf = Pi/Psum
    return Pf, perp_calc


def calc_p_vals(X, tol = 1e-5, perplexity = 30.0):
    (n, d) = X.shape

    partitions, pdist = calc_partitions(X, dist)
    P = [np.zeros((n, n)) for p in partitions]
    sigmas = [np.ones((n,1)) for p in partitions]

    for p in range(len(partitions)):
        for i in range(n):
            scaled_perplexity = perplexity*math.sqrt((n/DATA_SIZE)) #Adjust perplexity based on partiton data size
            pdists = pdist[p]
            Di = pdists[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            Pf, perp_calc = perplex_helper(Di, sigmas[p][i])
            #perplexity calculation - binary search for best param
            min_ = -np.inf
            max_ = np.inf
            count = 0
            diff = perp_calc - np.log(scaled_perplexity)
            while diff > tol and count < 50:
                if diff > 0:
                    min_ = sigmas[p][i].copy()
                    if max_ == np.inf or max_ == -np.inf:
                        sigmas[p][i] = sigmas[p][i] * 2.
                    else:
                        sigmas[p][i] = (sigmas[p][i] + max_) / 2.
                else:
                    max_ = sigmas[p][i].copy()
                    if min_ == np.inf or min_ == -np.inf:
                        sigmas[p][i] = sigmas[p][i] / 2.
                    else:
                        sigmas[p][i] = (sigmas[p][i] + min_) / 2.
                count+=1
                Pf, perp_calc = perplex_helper(Di, sigmas[p][i])
                diff = perp_calc - np.log(scaled_perplexity)

            P[p][i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pf
    return P, partitions

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

######################################################

###################### MAIN #########################

if __name__ == '__main__':

    desired_classes = [0,1,2]
    random_state = 1000
    df = pd.DataFrame(np.load("./art_4_8_100_300ptx.npy"))
    df['class'] = np.load('./art_4_8_100_300pty.npy')
    df = df.loc[df['class'].isin(desired_classes)]
    df = df.sample(n=DATA_SIZE, random_state=random_state)
    labels = np.array(df['class'])
    x_init = np.mat(df.drop('class', axis=1))

    np.random.seed(random_state)

    #desired_classes = [0,1,2,3,4]
    #x_init, labels = datasets.make_blobs(n_samples=DATA_SIZE,n_features=100, centers=5, cluster_std = 1)
    y = np.random.rand(DATA_SIZE, 2)

    #PCA and normalize into ball of radius 1
    x = decomposition.PCA(n_components=30).fit_transform(x_init)
    x = x/np.var(np.linalg.norm(x, axis=1))

    (n,d) = x.shape

    pvals, partitions = calc_p_vals(x)
    for idx in range(len(pvals)):
        pvals[idx] = pvals[idx] + pvals[idx].T       # calculate symmetric pvals
        pvals[idx] = pvals[idx] / np.sum(pvals[idx])

        #early exaggeration
        pvals[idx] = pvals[idx] * 4.
        pvals[idx] = np.maximum(pvals[idx], 1e-12)

    print("dumping pvals and partitions")

    with open('pval_art_4_8_100_300', 'wb') as f:
        pickle.dump(pvals, f)
    with open('partitions_art_4_8_100_300', 'wb') as f:
        pickle.dump(partitions, f)
    #gradient descent params
 #   weights = [1]+[1/len(partitions) for x in range(len(partitions)-1)]
    weights = get_dendrogram_weights(x)[::-1]
    with open('weights_art_4_8_100_300', 'wb') as f:
        pickle.dump(weights, f)

    dys = [np.zeros((n,2)) for x in range(len(partitions))]
    lr = 50

    # params for van der maaten's momentum magic!
    prev_y = np.zeros((n,2))
    init_momentum = 0.5
    st_momentum = 0.8
    gains = np.ones((n, 2))
    min_gain = 0.01
    eta = 500


    for iter in range(ITERATIONS):

        #compute qvals for each partition
        qdists = calc_dists(y, partitions, dist)
        qvals = []
        for d in qdists:
            num = 1. / (1. + d)
            num[range(n), range(n)] = 0.
            qval = num / np.sum(num)
            qval = np.maximum(qval, 1e-12)
            qvals.append(qval)

        #compute gradient -> computation trick again stolen from van der maaten
        for d in range(len(pvals)):
            PQ = pvals[d] - qvals[d]
            for i in range(n):
                dys[d][i,:] = weights[d]*np.sum(np.tile(PQ[:, i] * num[:, i], (2, 1)).T * (y[i, :] - y), 0)

        # van der maaten's momentum magic
        # if iter < 20:
        #     momentum = init_momentum
        # else:
        #     momentum = st_momentum
        # gains = (gains + 0.2) * ((dy > 0.) != (iy > 0.)) + \
        #         (gains * 0.8) * ((dy > 0.) == (iy > 0.))
        # gains[gains < min_gain] = min_gain
        # iy = momentum * iy - eta * (gains * dy)
        # y = y + iy
        # y = y - np.tile(np.mean(y, 0), (n, 1))

        #our shitty gd implementation
        y = y-lr*sum(dys)

        #print loss value
        if (iter + 1) % 10 == 0:
            L = 0
            for i in range(len(pvals)):
                L += np.sum(pvals[i] * np.log(pvals[i] / qvals[i]))  #KL divergence
            print("Iteration %d: error is %f" % (iter + 1, L))

        #stop early exaggeration
        if iter == int(0 / 5):
            for i in range(len(pvals)):
                pvals[i] = pvals[i]/4

        # if iter % 20 == 0:
        #     plt.figure()
        #     base = {}
        #     for class_ in desired_classes:
        #         base[class_] = []
        #     _dict = copy.deepcopy(base)
        #     for i,row in enumerate(y):
        #         _dict[labels[i]].append(row)
        #     for i in _dict.keys():
        #         plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
        #     plt.legend(desired_classes)
        #     plt.axis('equal')
        #     plt.savefig('diagram/iter'+str(iter))

    print("saving result")
    try:
        np.save('htsne_data_art_4_8_100_300', y)
        np.save('htsne_labels_art_4_8_100_300', labels)
        np.save('htsne_data_hd_art_4_8_100_300', x_init)
        
        plt.figure()
        base = {}
        for class_ in desired_classes:
            base[class_] = []
        _dict = copy.deepcopy(base)
        for i,row in enumerate(y):
            _dict[labels[i]].append(row)
        for i in _dict.keys():
            plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
        plt.legend(desired_classes)
    except Exception as e:
        print(e)
        plt.scatter(y[:,0], y[:,1], alpha=0.2)
    plt.axis('equal')
    plt.savefig('art_4_8_100_300.png')
    plt.show()

#####################################################
