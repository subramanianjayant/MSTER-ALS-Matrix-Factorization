import numpy as np
import scipy

np.random.seed(1000)

class MFConfig:
    def __init__(self, M = 'rand', d=2, n=20, m=50):

        self.d = d #latent space dimensionality

        if M=='rand':
            self.n = n
            self.m = m
            # self.M = np.mat(np.random.rand(self.n,self.m)) #ratings matrix
            self.M = np.mat(scipy.sparse.random(n,m, density = 0.1).A)

        else:
            self.M = M #if matrix is passed, we set it to M
            self.n = M.shape[0]
            self.m = M.shape[1]

        self.k_A = 2 #intended number of clusters in latent space
        self.k_B = 3

        self.A = np.mat(np.random.rand(self.n,self.d)) #initializations
        self.B = np.mat(np.random.rand(self.d,self.m))

        self.lr = 0.005 #learning rate
        self.lr_decay = 1e-6 #learning rate decay

        self.lambda_ = 0.5 #regularization for MSTER term in loss
        self.lambda_decay = 0 #decay for lambda

        self.eta = 0
        self.eta_decay = 0

        self.num_epochs = 200 #number of epochs for gradient descent

    def dump(self):
        return (self.M, self.A, self.B, self.k_A, self.k_B, self.lr,
                    self.lr_decay, self.lambda_, self.lambda_decay,
                    self.eta, self.eta_decay,
                    self.num_epochs)
