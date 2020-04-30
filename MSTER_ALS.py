import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from matplotlib import pyplot as plt

#dimensions
n = 100
m = 50
d = 2

#number of desired clusters for A and B
k_A = 2
k_B = 2

#initialization
M = np.mat(np.random.rand(n,m))
A = np.mat(np.random.rand(n,d))
B = np.mat(np.random.rand(d,m))

#calculate MSTER and find corresponding edge indices
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

def loss(M, A, B, rA, rB, lambda_):
    return np.linalg.norm(M-A*B, ord = 'fro')**2 - lambda_*(rA)#+rB)

def grad(A, vert):
    mat = np.mat(np.zeros((A.shape)))
    if vert[0,0]==vert[1,0]:
        for j in range(mat.shape[1]):
            mat[vert[0,0],j] = ((2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:]))) -
                                2*(A[vert[0,0],j]-A[vert[1,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))**2))
            mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[0,0],:]-A[vert[1,1],:])))
            mat[vert[1,1],j] = 2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))/((A[vert[0,0],j]-A[vert[1,1],j])**3)

    elif vert[0,0]==vert[1,1]:
        for j in range(mat.shape[1]):
            mat[vert[0,0],j] = ((2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:]))) +
                                2*(A[vert[1,0],j]-A[vert[0,0],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))**2))
            mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[0,0],:])))
            mat[vert[1,0],j] = -2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))/((A[vert[1,0],j]-A[vert[0,0],j])**3)

    elif vert[0,1]==vert[1,0]:
        for j in range(mat.shape[1]):
            mat[vert[0,1],j] = ((-2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:]))) -
                                2*(A[vert[0,1],j]-A[vert[1,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))**2))
            mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[0,1],:]-A[vert[1,1],:])))
            mat[vert[1,1],j] = 2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))/((A[vert[0,1],j]-A[vert[1,1],j])**3)

    elif vert[0,1]==vert[1,1]:
        for j in range(mat.shape[1]):
            mat[vert[0,1],j] = ((-2*(A[vert[0,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:]))) +
                                2*(A[vert[1,0],j]-A[vert[0,1],j])*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))) /
                                ((np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))**2))
            mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[0,1],:])))
            mat[vert[1,0],j] = -2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))/((A[vert[1,0],j]-A[vert[0,1],j])**3)

    else:
        for j in range(mat.shape[1]):
            mat[vert[0,0],j] = 2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))
            mat[vert[0,1],j] = -2*(A[vert[0,0],j]-A[vert[0,1],j])/(np.sum(np.square(A[vert[1,0],:]-A[vert[1,1],:])))
            mat[vert[1,0],j] = -2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))/((A[vert[1,0],j]-A[vert[1,1],j])**3)
            mat[vert[1,1],j] = 2*(np.sum(np.square(A[vert[0,0],:]-A[vert[0,1],:])))/((A[vert[1,0],j]-A[vert[1,1],j])**3)
    #print(mat)
    return mat


def train(lr = 0.01, lambda_ = 200, num_epochs = 200):
    global A
    global B
    global M
    global k_A
    global k_B
    for epoch in range(num_epochs):
        ratio_A, vertices_A = MSTER(A, k_A)
        ratio_B, vertices_B = MSTER(B.T, k_B)
        loss_ = loss(M,A,B,ratio_A, ratio_B, lambda_)
        if (epoch%10)<5:
            A = A + lr*((M-A*B)*B.T - lambda_*grad(A, vertices_A))
        else:
            B = B + lr*(A.T*(M-A*B) )#- lambda_*grad(B.T, vertices_B).T)
        #print("epoch {0} --- \t loss: {1}".format(epoch, loss_))
    return A,B

if __name__ == '__main__':
    lr = 0.01
    lambda_ = 0.001
    epochs = 200
    Ahat, Bhat = train(lr,lambda_, epochs)
    plt.figure(1)
    plt.scatter([A[:,0]], [A[:,1]])
    plt.figure(2)
    plt.scatter([B[0]], [B[1]])
    plt.title('Data in Latent Space (A,B)')
    plt.show()
