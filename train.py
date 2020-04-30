import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from matplotlib import pyplot as plt
from optimize.functions import MSTER, loss, grad
from config import MFConfig

M, A, B, k_A, k_B, lr, lambda_, num_epochs = MFConfig().dump()

def train():
    A_best = 0
    B_best = 0

    global M, A, B, k_A, k_B, lr, lambda_, num_epochs

    ratio_A, vertices_A = MSTER(A, k_A)
    ratio_B, vertices_B = MSTER(B.T, k_B)
    loss_prev = loss(M,A,B,ratio_A, ratio_B, lambda_)
    for epoch in range(num_epochs):
        if (epoch%10)<5:
            A = A - lr*(-(M-A*B)*B.T - lambda_*grad(A, vertices_A))
        else:
            B = B - lr*(-A.T*(M-A*B) - lambda_*grad(B.T, vertices_B).T)

        ratio_A, vertices_A = MSTER(A, k_A)
        ratio_B, vertices_B = MSTER(B.T, k_B)
        loss_ = loss(M,A,B,ratio_A, ratio_B, lambda_)
        print("epoch {0} --- \t loss: {1} \t MSTER: {2}".format(epoch+1,
                loss_, ratio_A))

        if loss_<loss_prev:
            A_best = A.copy()
            B_best = B.copy()
        loss_prev = loss_

    return A_best,B_best #returns best model in terms of loss

if __name__ == '__main__':
    A_best, B_best = train()
    plt.figure(1)
    plt.scatter([A_best[:,0]], [A_best[:,1]], color='blue')
    #plt.figure(2)
    plt.scatter([B_best[0]], [B_best[1]], color='orange')
    plt.title('Data in Latent Space')
    plt.legend(['A data','B data'])
    plt.show()
