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

        self.k = 5 #intended number of clusters in latent space

        # RANDOM INITIALIZATION
        self.P = 10*np.mat(np.random.rand(self.n,self.d)) #initializations

        # PCA INITIALIZATION
        # pca = PCA(n_components=self.d)
        # pca.fit_transform(self.M)
        # self.P = np.mat(pca.components_)

        print("P: {}".format(self.P.shape))

        self.lr = 1e-2 #learning rate
        self.lr_decay = 0 #learning rate decay

        self.num_epochs = 500 #number of epochs for gradient descent

        self.clip = 1000

    def dump(self):
        return (self.M, self.P, self.k, self.lr,
                self.lr_decay, self.num_epochs, self.clip)
