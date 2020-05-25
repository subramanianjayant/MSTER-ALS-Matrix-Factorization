import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from matplotlib import pyplot as plt
from optimize.functions import MSTER, loss, grad, H
from config import MFConfig
import pandas as pd

df = pd.read_csv('mnist_784_zip/data/mnist_784_csv.csv').sample(n=1000, random_state=1600)
labels = np.array(df['class'])
data = np.mat(df.drop('class', axis=1))
(M, A, B, k_A, k_B, lr, lr_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs) = MFConfig(M=data).dump()
pca_init = A.copy()

def train():
    global M, A, B, k_A, k_B, lr, lr_decay, lambda_, lambda_decay, eta, eta_decay, num_epochs
    A_best = A.copy()
    B_best = B.copy()

    ratio_A, vertices_A = MSTER(A, k_A)
    ratio_B, vertices_B = MSTER(B.T, k_B)
    loss_best = loss(M,A,B,ratio_A, ratio_B, lambda_, eta)
    for epoch in range(num_epochs):
        best = ''
        if (epoch%10)<5:
            A = A - lr*(-(M-A*B)*B.T - lambda_*grad(A, vertices_A) - eta*(2*H(M.shape[0]).T*H(M.shape[0])*A))
        else:
            B = B - lr*(-A.T*(M-A*B)) #- lambda_*grad(B.T, vertices_B).T - eta*(2*B*H(M.shape[1]).T*H(M.shape[1])))

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

    _dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    for i,row in enumerate(np.array(A_best)):
        _dict[labels[i]].append(row)

    plt.figure(1)
    plt.title('d=2 MSTER-ALS Representation of MNIST Sample (n=1000)')
    for i in range(k_A):
        plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.3)
    plt.legend(range(10))

    _dict2 = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    for i,row in enumerate(np.array(pca_init)):
        _dict2[labels[i]].append(row)

    plt.figure(2)
    plt.title('d=2 PCA Representation of MNIST Sample (n=1000)')
    for i in range(k_A):
        plt.scatter(np.array(_dict2[i])[:,0], np.array(_dict2[i])[:,1], alpha=0.3)
    plt.legend(range(10))

    plt.show()
