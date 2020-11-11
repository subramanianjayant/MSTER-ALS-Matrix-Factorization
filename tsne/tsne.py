import math
import numpy as np
import pandas as pd
from sklearn import datasets, decomposition, metrics
from matplotlib import pyplot as plt
from hierarchy import dist, merge, calc_partitions, calc_dists
import copy

############# CONSTANTS ##############

DATA_SIZE = 100
ITERATIONS = 300

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
    pdists = pdist[0]

    P = np.zeros((n,n))
    sigmas = np.ones((n,1))


    for i in range(n):
        Di = pdists[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        Pf, perp_calc = perplex_helper(Di, sigmas[i])
        #perplexity calculation - binary search for best param
        min_ = -np.inf
        max_ = np.inf
        count = 0
        diff = perp_calc - np.log(perplexity)
        while diff > tol and count < 50:
            if diff > 0:
                min_ = sigmas[i].copy()
                if max_ == np.inf or max_ == -np.inf:
                    sigmas[i] = sigmas[i] * 2.
                else:
                    sigmas[i] = (sigmas[i] + max_) / 2.
            else:
                max_ = sigmas[i].copy()
                if min_ == np.inf or min_ == -np.inf:
                    sigmas[i] = sigmas[i] / 2.
                else:
                    sigmas[i] = (sigmas[i] + min_) / 2.
            count+=1
            Pf, perp_calc = perplex_helper(Di, sigmas[i])
            diff = perp_calc - np.log(perplexity)

        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pf
    return P

######################################################

###################### MAIN #########################

if __name__ == '__main__':

    desired_classes = [0,1,3,6,8]
    random_state = 1000
    df = pd.read_csv('../mnist_784_zip/data/mnist_784_csv.csv')
    df = df.loc[df['class'].isin(desired_classes)]
    df = df.sample(n=DATA_SIZE, random_state=random_state)
    labels = np.array(df['class'])
    x_init = np.mat(df.drop('class', axis=1))

    np.random.seed(random_state)

    # desired_classes = [0,1,2,3,4]
    # x_init, labels = datasets.make_blobs(n_samples=DATA_SIZE,n_features=100, centers=5, cluster_std = 1)
    y = np.random.rand(DATA_SIZE, 2)



    #PCA and normalize into ball of radius 1
    x = decomposition.PCA(n_components=30).fit_transform(x_init)
    x = x/np.var(np.linalg.norm(x, axis=1))

    (n,d) = x.shape

    pvals  = calc_p_vals(x)

    pvals = pvals + pvals.T       # calculate symmetric pvals
    pvals = pvals / np.sum(pvals)

    #early exaggeration
    pvals = pvals * 4.
    pvals = np.maximum(pvals, 1e-12)

    #gradient descent params
    dy = np.zeros((n,2))
    lr = 50

    # params for van der maaten's momentum magic!
    prev_y = np.zeros((n,2))
    init_momentum = 0.5
    st_momentum = 0.8
    gains = np.ones((n, 2))
    min_gain = 0.01
    eta = 500

    print('Preprocessing done...\n\n')
    for iter in range(ITERATIONS):


        #qvals for first layer -> computation trick from van der maaten's code
        sum_Y = np.sum(np.square(y), 1)
        num = -2. * np.dot(y, y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        qvals = num / np.sum(num)
        qvals = np.maximum(qvals, 1e-12)

        #compute gradient -> computation trick again stolen from van der maaten
        PQ = pvals - qvals
        for i in range(n):
            dy[i,:] = np.sum(np.tile(PQ[:, i] * num[:, i], (2, 1)).T * (y[i, :] - y), 0)

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
        y = y-lr*dy

        #print loss value
        if (iter + 1) % 10 == 0:
            L = np.sum(pvals * np.log(pvals / qvals))  #KL divergence
            print("Iteration %d: error is %f" % (iter + 1, L))

        #stop early exaggeration
        if iter == 100:
            pvals = pvals/4

    try:
        np.save('tsne_data.csv', y)
        np.save('tsne_labels.csv', labels)
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
    plt.show()

#####################################################
