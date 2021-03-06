import numpy as np
import scipy
from sklearn.decomposition import PCA

class MFConfig:
    def __init__(self, M = None, k=6, seed = 1600, d=2, n=100, m=20):

        self.d = d #latent space dimensionality
        self.random_state = seed
        np.random.seed(self.random_state)

        self.random_state = 1800
        np.random.seed(self.random_state)

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

        self.k = k #intended number of clusters in latent space

        # RANDOM INITIALIZATION
        # self.P = 3*np.mat(np.random.rand(self.m,self.d))

        # PCA INITIALIZATION
        pca = PCA(n_components=self.d)
        pca.fit_transform(self.M)
        self.P = np.mat(pca.components_).T

        print("P: {}".format(self.P.shape))

        self.lr = 10 #learning rate
        self.lr_decay = 0 #learning rate decay

        self.lambda_ = 0 #
        self.lambda_decay = 0

        self.num_epochs = 100 #number of epochs for gradient descent

        self.clip = 10000000

    def dump(self):
        return (self.M, self.P, self.k,
                self.lr, self.lr_decay,
                self.lambda_, self.lambda_decay,
                self.num_epochs, self.clip,
                self.random_state)
