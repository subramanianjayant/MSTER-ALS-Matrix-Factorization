import numpy as np
from sklearn.metrics import pairwise
import networkx as nx

#calculate MSTER and find corresponding edge indices
def H(n):
    return np.mat(np.identity(n)-(1/n)*np.ones((n,n)))

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
    for i in range(2,k+1):
        mst.remove_edge(edge_list[size - i][0], edge_list[size - i][1])
    clusters = nx.connected_components(mst)
    balance = 0
    for cluster in clusters:
        balance += (len(cluster) - size/k)**2
        print(balance)
    return ratio, vertices, balance

def loss(M, A, B, rA, bA, lambda_, eta):
    return (0.5*np.linalg.norm(M-A*B, ord = 'fro')**2
                - lambda_*(rA)#+rB)
                - eta*(np.trace(H(M.shape[0])*A*A.T*H(M.shape[0]).T)) #+ np.trace(H(M.shape[1]).T*B.T*B*H(M.shape[1]))))
                + vega*(bA))

def grad(A, vert):
    mat = np.mat(np.zeros((A.shape)))
    if vert[0,0]==vert[1,0]:
        for j in range(mat.shape[1]):
            mat[vert[0,0],j] = ((2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:]))) -
                                2*(A[vert[0,0],j]-A[vert[1,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))**2))
            mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))
            mat[vert[1,1],j] = (2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[0,0],j]-A[vert[1,1],j]) /
                                ((np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))**2))

    elif vert[0,0]==vert[1,1]:
        for j in range(mat.shape[1]):
            mat[vert[0,0],j] = ((2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:]))) +
                                2*(A[vert[1,0],j]-A[vert[0,0],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))**2))
            mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))
            mat[vert[1,0],j] = (-2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[0,0],j]) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))**2))

    elif vert[0,1]==vert[1,0]:
        for j in range(mat.shape[1]):
            mat[vert[0,1],j] = ((-2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:]))) -
                                2*(A[vert[0,1],j]-A[vert[1,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))**2))
            mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))
            mat[vert[1,1],j] = (2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[0,1],j]-A[vert[1,1],j]) /
                                ((np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))**2))

    elif vert[0,1]==vert[1,1]:
        for j in range(mat.shape[1]):
            mat[vert[0,1],j] = ((-2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:]))) +
                                2*(A[vert[1,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))**2))
            mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))
            mat[vert[1,0],j] = (-2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[0,1],j]) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))**2))

    else:
        for j in range(mat.shape[1]):
            mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))
            mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))
            mat[vert[1,0],j] = (-2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[1,1],j]) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))**2))
            mat[vert[1,1],j] = (2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))*(A[vert[1,0],j]-A[vert[1,1],j]) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))**2))
    #print(mat)
    return mat

def balance_grad():
    #Need something that is differentiable
    pass
