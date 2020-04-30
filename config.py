import numpy as np

class MFConfig:
    def __init__(self, M = 'rand', d=2, n=100, m=50):

        self.d = d #latent space dimensionality

        if M=='rand':
            self.n = n
            self.m = m
            self.M = np.mat(np.random.rand(self.n,self.m)) #ratings matrix

        else:
            self.M = M #if matrix is passed, we set it to M
            self.n = M.shape[0]
            self.m = M.shape[1]

        self.k_A = 2 #intended number of clusters in latent space
        self.k_B = 2

        self.A = np.mat(np.random.rand(self.n,self.d)) #initializations
        self.B = np.mat(np.random.rand(self.d,self.m))

        self.lr = 0.01 #learning rate
        self.lambda_ = 5 #regularization for MSTER term in loss
        self.num_epochs = 200 #number of epochs for gradient descent

    def dump(self):
        return self.M, self.A, self.B, self.k_A, self.k_B, self.lr, self.lambda_, self.num_epochs
