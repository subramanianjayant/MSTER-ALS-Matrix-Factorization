import numpy as np
import scipy
from sklearn.decomposition import PCA

np.random.seed(1601)

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

        self.k_A = 5 #intended number of clusters in latent space

        # RANDOM INITIALIZATIONS
        self.A = 10*np.mat(np.random.rand(self.n,self.d)) #initializations
        self.B = 3*np.mat(np.random.rand(self.d,self.m))

        # PCA INITIALIZATIONS
        # pca = PCA(n_components=self.d)
        # self.A = np.mat(pca.fit_transform(self.M))
        # self.B = np.mat(pca.components_)

        print("A: {}".format(self.A.shape))
        print("B: {}".format(self.B.shape))

        self.lr_a = 1e-2 #learning rate
        self.lr_a_decay = 0 #learning rate decay

        self.lr_b = 1e-4 #learning rate
        self.lr_b_decay = 0 #learning rate decay

        self.lambda_ = 20 #regularization for MSTER term in loss
        self.lambda_decay = 0 #decay for lambda

        self.eta = 0
        self.eta_decay = 0

        self.num_epochs = 500 #number of epochs for gradient descent

        self.clip_a = 1000
        self.clip_b = 1000

    def dump(self):
        return (self.M, self.A, self.B, self.k_A, self.lr_a,
                    self.lr_a_decay, self.lr_b,
                    self.lr_b_decay, self.lambda_, self.lambda_decay,
                    self.eta, self.eta_decay,
                    self.num_epochs, self.clip_a, self.clip_b)
