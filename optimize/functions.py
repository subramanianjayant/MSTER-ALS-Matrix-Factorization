import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

#calculate MSTER and find corresponding edge indices
def H(n):
    return np.mat(np.identity(n)-(1/n)*np.ones((n,n)))

#MST edge ratio
def MSTER(A, k):
    size = A.shape[0]
    assert k>1
    D = pairwise.euclidean_distances(A)
    cg = nx.from_numpy_matrix(D)
    mst = nx.minimum_spanning_tree(cg)
    edge_list = list(mst.edges(data = True))
    edge_list.sort(key=lambda x: x[2]['weight'])
    ratio = (edge_list[size-k][2]['weight']/edge_list[size-k-1][2]['weight'])**2-1
    vertices = np.array([[edge_list[size-k][0], edge_list[size-k][1]],[edge_list[size-k-1][0],edge_list[size-k-1][1]]])
    return ratio, vertices

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

def pairwise_distance(A):
    pass

def E(n, i):
    mat = np.zeros((n,n))
    mat[i,i] = 1
    return np.mat()

def loss(ratio):
    return ratio

# def grad(A, vert):
#     mat = np.mat(np.zeros((A.shape)))
#     if vert[0,0]==vert[1,0]:
#         for j in range(mat.shape[1]):
#             mat[vert[0,0],j] = ((2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:]))) -
#                                 2*(A[vert[0,0],j]-A[vert[1,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
#                                 ((np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))**2))
#             mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))
#             mat[vert[1,1],j] = (2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[0,0],j]-A[vert[1,1],j]) /
#                                 ((np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))**2))
#
#     elif vert[0,0]==vert[1,1]:
#         for j in range(mat.shape[1]):
#             mat[vert[0,0],j] = ((2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:]))) +
#                                 2*(A[vert[1,0],j]-A[vert[0,0],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
#                                 ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))**2))
#             mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))
#             mat[vert[1,0],j] = (-2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[0,0],j]) /
#                                 ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))**2))
#
#     elif vert[0,1]==vert[1,0]:
#         for j in range(mat.shape[1]):
#             mat[vert[0,1],j] = ((-2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:]))) -
#                                 2*(A[vert[0,1],j]-A[vert[1,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
#                                 ((np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))**2))
#             mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))
#             mat[vert[1,1],j] = (2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[0,1],j]-A[vert[1,1],j]) /
#                                 ((np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))**2))
#
#     elif vert[0,1]==vert[1,1]:
#         for j in range(mat.shape[1]):
#             mat[vert[0,1],j] = ((-2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:]))) +
#                                 2*(A[vert[1,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
#                                 ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))**2))
#             mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))
#             mat[vert[1,0],j] = (-2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[0,1],j]) /
#                                 ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))**2))
#
#     else:
#         for j in range(mat.shape[1]):
#             mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))
#             mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))
#             mat[vert[1,0],j] = (-2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[1,1],j]) /
#                                 ((np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))**2))
#             mat[vert[1,1],j] = (2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[1,1],j]) /
#                                 ((np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))**2))
#     #print(mat)
#     return mat

def grad(M, P, A_, B_, C_, D_):
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
    return coeff*mat
