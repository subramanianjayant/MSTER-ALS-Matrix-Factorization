import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from matplotlib import pyplot as plt
from optimize.functions import MSTER, loss, grad, H
from config import MFConfig

(M, A, B, k_A, k_B, lr, lr_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs) = MFConfig().dump()

def train():
    global M, A, B, k_A, k_B, lr, lr_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs
    A_best = 0
    B_best = 0

    ratio_A, vertices_A = MSTER(A, k_A)
    ratio_B, vertices_B = MSTER(B.T, k_B)
    loss_best = loss(M,A,B,ratio_A, ratio_B, lambda_, eta)
    for epoch in range(num_epochs):
        best = ''
        if (epoch%10)<5:
            A = A - lr*(-(M-A*B)*B.T - lambda_*grad(A, vertices_A) - eta*(2*H(M.shape[0]).T*H(M.shape[0])*A))
        else:
            B = B - lr*(-A.T*(M-A*B) )# - lambda_*grad(B.T, vertices_B).T - eta*(2*B*H(M.shape[1]).T*H(M.shape[1])))

        ratio_A, vertices_A = MSTER(A, k_A)
        ratio_B, vertices_B = MSTER(B.T, k_B)
        loss_ = loss(M,A,B,ratio_A, ratio_B, lambda_, eta)

        if loss_<loss_best:
            A_best = A.copy()
            B_best = B.copy()
            best = 'best'
            loss_best = loss_

        lr -= lr_decay
        lambda_ -= lambda_decay
        eta -= eta_decay

        print("epoch {0} --- \t loss: {1} \t norm contribution: {2} \t {3}".format(epoch,
                loss_, np.linalg.norm(M-A*B, ord = 'fro')**2, best))

    return A_best,B_best #returns best model in terms of loss

if __name__ == '__main__':
    A_best, B_best = train()
    plt.figure(1)
    plt.scatter([A_best[:,0]], [A_best[:,1]], color='blue',alpha=0.4)
    plt.scatter([B_best[0]], [B_best[1]], color='orange',alpha=0.4)
    plt.title('Data in Latent Space (Best Model)')
    plt.legend(['A data','B data'])

    # plt.figure(2)
    # plt.scatter([A[:,0]], [A[:,1]], color='green',alpha=0.4)
    # plt.scatter([B[0]], [B[1]], color='purple', alpha=0.4)
    # plt.title('Data in Latent Space')
    # plt.legend(['A data','B data'])

    plt.show()
