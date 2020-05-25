import numpy as np
import scipy
from sklearn.decomposition import PCA

np.random.seed(1600)

class MFConfig:
    def __init__(self, M = None, d=2, n=20, m=50):

        self.d = d #latent space dimensionality

        if M is None:
            self.n = n
            self.m = m
            # self.M = np.mat(np.random.rand(self.n,self.m)) #ratings matrix
            self.M = np.mat(scipy.sparse.random(n,m, density = 0.1).A)

        else:
            self.M = M #if matrix is passed, we set it to M
            self.n = M.shape[0]
            self.m = M.shape[1]

        print("M: {}".format(self.M.shape))

        self.k_A = 10 #intended number of clusters in latent space
        self.k_B = 4

        #self.A = np.mat(np.random.rand(self.n,self.d)) #initializations
        self.B = np.mat(np.random.rand(self.d,self.m))
        pca = PCA(n_components=self.d)
        self.A = np.mat(pca.fit_transform(self.M))
        print("A: {}".format(self.A.shape))
        print("B: {}".format(self.B.shape))

        self.lr = 0.001 #learning rate
        self.lr_decay = 1e-6 #learning rate decay

        self.lambda_ = 5000 #regularization for MSTER term in loss
        self.lambda_decay = 0 #decay for lambda

        self.eta = 0
        self.eta_decay = 0

        self.num_epochs = 50 #number of epochs for gradient descent

    def dump(self):
        return (self.M, self.A, self.B, self.k_A, self.k_B, self.lr,
                    self.lr_decay, self.lambda_, self.lambda_decay,
                    self.eta, self.eta_decay,
                    self.num_epochs)
