import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

def H(n):
    return np.mat(np.identity(n)-(1/n)*np.ones((n,n)))

def calc_objective(P, M, k, lambda_):
    P = np.mat(P.reshape(M.shape[1],-1))
    ratio, _, _, _, _ = LCR(M*P,k)
    return -1*(ratio - lambda_* loss_distance(M, P))

#Linkage Cost Ratio (Ward)
def LCR(A, k):
    agc = AgglomerativeClustering(n_clusters=k+1, linkage='ward').fit(A)
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

    temp_vals['new'] = temp_vals[C_index]+temp_vals[D_index]
    temp_vals = temp_vals.drop([C_index, D_index], axis=1)
    temp_vals = temp_vals.rename(columns={x:y for x,y in zip(temp_vals.columns,range(0,len(temp_vals.columns)))})
    #print(temp_vals)
    min = np.inf
    A_index = 0
    B_index = 0
    for i in range(len(temp_vals.columns)):
        for j in range(i+1,len(temp_vals.columns)):
            P_ = np.array(temp_vals[i])
            Q_ = np.array(temp_vals[j])
            cost = MergeCost(A, P_, Q_)/(sum(P_)*sum(Q_))
            if  cost < min:
                min = cost
                A_index = i
                B_index = j
    A_ = np.array(temp_vals[A_index])
    B_ = np.array(temp_vals[B_index])
    # ratio = (sum(C_)*sum(D_)*MergeCost(A, A_, B_))/(sum(A_)*sum(B_)*MergeCost(A, C_, D_)) #balanced and imbalanced clusters treated the same
    ratio = (4*sum(C_)*sum(D_)*MergeCost(A, A_, B_))/(((sum(A_)+sum(B_))**2)*MergeCost(A, C_, D_)) #favor balanced clusters
    return ratio, A_, B_, C_, D_

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
    return 1e5 * np.mat(np.mat(A) / np.trace(np.mat(A).T * np.mat(A)))

def loss(M, P, ratio, lambda_):
    print(ratio, loss_distance(M,P))
    return ratio - lambda_* loss_distance(M, P)

def loss_distance(M, P):
    A = M*P
    n = M.shape[0]
    return np.linalg.norm(distance_penalty(pairwise_square_distance(M)) - pairwise_square_distance(A), ord = 'fro')**2 / (n * (n-1))

def distance_penalty(D): #function to penalize distances by
    return np.exp(-1*D)

def grad(M, P, A_, B_, C_, D_, lambda_):
    A = M*P
    mat = np.mat(np.zeros((A.shape)))
    # coeff = (sum(C_)*sum(D_))/(sum(A_)*sum(B_)) #balanced and imbalanced clusters treated the same
    coeff = (4*sum(C_)*sum(D_))/((sum(A_)+sum(B_))**2) #favor balanced clusters over imbalanced
    A_p = np.diag(A_)
    B_p = np.diag(B_)
    C_p = np.diag(C_)
    D_p = np.diag(D_)
    n = len(A)
    temp1 = MergeCost(A, C_, D_)
    temp2 = MergeCost(A, A_, B_)
    diff1 = 2*(sum(B_)*M.T*A_p*A_p.T*M+sum(A_)*M.T*B_p*B_p.T*M-M.T*B_p*np.ones((n,n))*A_p*M-M.T*A_p*np.ones((n,n))*B_p*M)*P
    diff2 = 2*(sum(D_)*M.T*C_p*C_p.T*M+sum(C_)*M.T*D_p*D_p.T*M-M.T*D_p*np.ones((n,n))*C_p*M-M.T*C_p*np.ones((n,n))*D_p*M)*P
    mat = (temp1*diff1-temp2*diff2)/(temp2**2)
    print(coeff*mat)
    print(grad_distance(M,P))
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
