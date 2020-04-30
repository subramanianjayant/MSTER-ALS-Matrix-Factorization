import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from matplotlib import pyplot as plt
from optimize.functions import MSTER, loss, grad
from config import MFConfig

M, A, B, k_A, k_B, lr, lambda_, num_epochs = MFConfig().dump()

def train():
    global M, A, B, k_A, k_B, lr, lambda_, num_epochs

    for epoch in range(num_epochs):
        ratio_A, vertices_A = MSTER(A, k_A)
        ratio_B, vertices_B = MSTER(B.T, k_B)
        loss_ = loss(M,A,B,ratio_A, ratio_B, lambda_)
        if (epoch%10)<5:
            A = A + lr*((M-A*B)*B.T - lambda_*grad(A, vertices_A))
        else:
            B = B + lr*(A.T*(M-A*B) )#- lambda_*grad(B.T, vertices_B).T)
        print("epoch {0} --- \t loss: {1}".format(epoch, loss_))
    return A,B

if __name__ == '__main__':
    Ahat, Bhat = train()
    plt.figure(1)
    plt.scatter([Ahat[:,0]], [Ahat[:,1]], color='blue')
    plt.scatter([Bhat[0]], [Bhat[1]], color='orange')
    plt.title('Data in Latent Space')
    plt.legend(['A data','B data'])
    plt.show()
