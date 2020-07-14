import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def adam(args, num_epochs = 100, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    (P, M, k, lambda_) = args
    m = np.mat(np.zeros(P.shape))
    v = np.mat(np.zeros(P.shape))
    t = 0
    P_best = P
    loss_best = -np.inf
    while t < num_epochs:
        best = ''
        t += 1
        ratio, C_, D_ = LCR(M*P, k)
        loss_ = loss(M, P, ratio, lambda_)
        if loss_ > loss_best:
            P_best = P
            loss_best = loss_
            best = 'best'
        print("epoch {0} --- \t loss: {1} --- \t LCR: {2} \t {3}".format(t, loss_, ratio, best))
        g = grad(M, P, C_, D_, lambda_)
        m = beta1*m+(1-beta1)*g
        v = beta2*v+(1-beta2)*np.square(g)
        m_hat = m/(1-(beta1**t))
        v_hat = v/(1-(beta2**t))
        P = P + np.divide(alpha*m_hat, np.sqrt(v_hat)+epsilon)
    return P_best # for highest loss result
    # return P # for latest result

def VGA(args, num_epochs, lr = 0.01):
    (P, M, k, lambda_) = args
    t = 0
    P_best = P
    loss_best = -np.inf
    while t < num_epochs:
        best = ''
        t += 1
        ratio, C_, D_ = LCR(M*P, k)
        loss_ = loss(M, P, ratio, lambda_)
        if loss_ > loss_best:
            P_best = P
            loss_best = loss_
            best = 'best'
        print("epoch {0} --- \t loss: {1} --- \t LCR: {2} \t {3}".format(t, loss_, ratio, best))
        g = grad(M, P, C_, D_, lambda_)
        P = P + lr*g
    return P_best # for highest loss result
    # return P # for latest result

def H(n):
    return np.mat(np.identity(n)-(1/n)*np.ones((n,n)))

def calc_objective(P, M, k, lambda_):
    P = np.mat(P.reshape(M.shape[1],-1))
    ratio, _, _, _, _ = LCR(M*P,k)
    return -1*(ratio - lambda_* loss_distance(M, P))

#Linkage Cost Ratio (Ward)
def LCR(A, k):
    agc = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(A)
    temp_vals = pd.get_dummies(pd.Series(agc.labels_))
    min = np.inf
    C_index = 0
    D_index = 0
    for i in range(len(temp_vals.columns)):
        for j in range(i+1,len(temp_vals.columns)):
            P_ = np.array(temp_vals[i])
            Q_ = np.array(temp_vals[j])
            cost = MergeCost(A, P_, Q_)/(sum(P_)*sum(Q_))
            if  cost < min:
                min = cost
                C_index = i
                D_index = j
    C_ = np.array(temp_vals[C_index])
    D_ = np.array(temp_vals[D_index])
    ratio = MergeCost(A, C_, D_)/(np.trace(A*A.T)) #favor balanced clusters
    return ratio, C_, D_

def MergeCost(A, A_, B_):
    A_p = np.diag(A_)
    B_p = np.diag(B_)
    return np.asscalar(np.trace(B_p)*np.trace(A_p*A*A.T*A_p)+np.trace(A_p)*np.trace(B_p*A*A.T*B_p)-2*(A.T*A_p*np.mat(np.ones((len(A),1)))).T*(A.T*B_p*np.mat(np.ones((len(A),1)))))

def pairwise_square_distance(A):
    n = A.shape[0]
    diag = np.zeros((n,n))
    for i in range(n):
        diag += (E(n,i) * A * A.T * E(n,i))
    return np.mat(diag*np.mat(np.ones((n,n))) + (diag*np.mat(np.ones((n,n)))).T - 2*A*A.T)

def E(n, i):
    mat = np.zeros((n,n))
    mat[i,i] = 1
    return np.mat(mat)

def normalise(A):
    #divide by trace of covariance matrix
    return 1e5*np.divide(np.mat(A), np.trace(np.mat(A) * np.mat(A).T))

def loss(M, P, ratio, lambda_):
    #print(ratio, loss_distance(M,P))
    return ratio - lambda_* loss_distance(M, P)

def loss_distance(M, P):
    A = M*P
    n = M.shape[0]
    return np.linalg.norm(distance_penalty(pairwise_square_distance(M)) - pairwise_square_distance(A), ord = 'fro')**2 / (n * (n-1))

def distance_penalty(D): #function to penalize distances by
    # return D
    return np.exp(-1*D)

def grad(M, P, C_, D_, lambda_):
    A = M*P
    mat = np.mat(np.zeros((A.shape)))
    coeff = 1 #favor balanced clusters over imbalanced
    C_p = np.diag(C_)
    D_p = np.diag(D_)
    n = len(A)
    temp1 = np.trace(A*A.T)
    temp2 = MergeCost(A, C_, D_)
    diff1 = 2*(sum(D_)*M.T*C_p*C_p.T*M+sum(C_)*M.T*D_p*D_p.T*M-M.T*D_p*np.ones((n,n))*C_p*M-M.T*C_p*np.ones((n,n))*D_p*M)*P
    diff2 = 2*M.T*M*P
    mat = (temp1*diff1-temp2*diff2)/(temp2**2)
    return np.mat(coeff*mat - lambda_*grad_distance(M, P))

def grad_distance(M, P):
    A = M*P
    n = M.shape[0]
    coeff =  -4 / (n * (n-1))
    sum1 = M.T * np.mat(np.diag(np.diag((distance_penalty(pairwise_square_distance(M)) - pairwise_square_distance(A)) * np.mat(np.ones((n,n)))))) * A
    sum2 = M.T * np.mat(np.diag(np.diag(np.mat(np.ones((n,n))) * (distance_penalty(pairwise_square_distance(M)) - pairwise_square_distance(A))))) * A
    term3 = 2 * M.T * (distance_penalty(pairwise_square_distance(M)) - pairwise_square_distance(A)) * A
    grad = coeff * (sum1 + sum2 - term3)
    return grad

# takes in a set of points and returns how pure the neirest neighbourhoods are; scaled from 0 (completely wrong) to 1 (completely correct)
## args: np.mat, array/np.arr
def nn_score(data, labels):
    n = data.shape[0]
    # set k as 10% of all the points
    k = int(n/10)
    score = 0
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(data)

    score = 0
    for i in range(n):
        local_neigh = neigh.kneighbors(data[i], k, return_distance = False)
        for neighbor in local_neigh[0]:
            if labels[i] == labels[neighbor]:
                score += 1

    return score/(n * k)
