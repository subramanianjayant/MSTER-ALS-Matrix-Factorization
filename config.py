import numpy as np
import scipy
from sklearn.decomposition import PCA
import pandas as pd

class MFConfig:
    def __init__(self, dataset = None, desired_classes = [], k=6, num_points=200, seed = 1600, d=2):

        self.d = d #latent space dimensionality
        self.random_state = seed
        np.random.seed(self.random_state)


        if dataset == 'MNIST':
            df = pd.read_csv('mnist_784_zip/data/mnist_784_csv.csv')
            df = df.loc[df['class'].isin(desired_classes)]
            df = df.sample(n=num_points, random_state=seed)
            labels = np.array(df['class'])
            data = np.mat(df.drop('class', axis=1))
            pca_0 = PCA(n_components = 30) #initial dim reduction for faster MST computation (from tSNE paper)
            init_data = np.mat(pca_0.fit_transform(data))
            synth_pca = PCA(n_components = 2)
            pca_init = np.mat(synth_pca.fit_transform(init_data))
            self.M = init_data
            self.pca_init = pca_init
            self.labels = labels

        elif dataset == 'WINE':
            df = pd.read_csv('wine.data', header = None)
            df = df.loc[df[0].isin(desired_classes)]
            if num_points < len(df):
                df = df.sample(n=num_points, random_state=seed)
            labels = np.array(df[0])
            data = np.mat(df.drop(0, axis=1))
            synth_pca = PCA(n_components = 2)
            pca_init = np.mat(synth_pca.fit_transform(data))
            self.M = data
            self.pca_init = pca_init
            self.labels = labels

        elif dataset == 'RADAR':
            df = pd.read_csv('RADAR_mMDS_351d.csv', header = None)
            labels = pd.read_csv('radar_labels.csv', header = None)
            df['labels'] = np.array(labels[1])
            df = df.loc[df['labels'].isin(desired_classes)]
            if num_points < len(df):
                df = df.sample(n=num_points, random_state=seed)
            labels = np.array(df['labels'])
            data = np.mat(df.drop('labels', axis=1))
            pca_0 = PCA(n_components = 30) #initial dim reduction for faster MST computation (from tSNE paper)
            init_data = np.mat(pca_0.fit_transform(data))
            pca_init = np.mat(synth_pca.fit_transform(init_data))
            self.M = init_data
            self.pca_init = pca_init
            self.labels = labels

        elif dataset == 'PROTEIN':
            df = pd.read_csv('nmMDS_131d.csv', header = None)
            labels = pd.read_csv('protein_labels.csv', header = None)
            df['labels'] = np.array(labels[1])
            df = df.loc[df['labels'].isin(desired_classes)]
            if num_points < len(df):
                df = df.sample(n=num_points, random_state=seed)
            labels = np.array(df['labels'])
            data = np.mat(df.drop('labels', axis=1))
            synth_pca = PCA(n_components = 2)
            pca_init = np.mat(synth_pca.fit_transform(data))
            self.M = data
            self.pca_init = pca_init
            self.labels = labels

        elif dataset == 'IRIS':
            df = pd.read_csv('iris.data', header = None)
            df = df.loc[df[4].isin(desired_classes)]
            if num_points < len(df):
                df = df.sample(n=num_points, random_state=seed)
            labels = np.array(df[4])
            data = np.mat(df.drop(4, axis=1))
            synth_pca = PCA(n_components=2)
            pca_init = synth_pca.fit_transform(data)
            self.M = data
            self.pca_init = pca_init
            self.labels = labels

        elif dataset == 'octahedron':
            num_dimensions = 20
            arr = np.zeros((6 ,num_dimensions))
            arr[0,0:3] = [5, 5, 0]
            arr[1,0:3] = [5, -5, 0]
            arr[2,0:3] = [-5, 5, 0]
            arr[3,0:3] = [-5,-5, 0]
            arr[4,0:3] = [0, 0, 5]
            arr[5,0:3] = [0, 0, -5]
            init_data, labels = datasets.make_blobs(n_samples=num_points, n_features=num_dimensions, centers = arr, cluster_std=0.6)
            init_data = np.mat(init_data)
            synth_pca = PCA(n_components=2)
            pca_init = synth_pca.fit_transform(init_data)
            self.M = init_data
            self.pca_init = pca_init
            self.labels = labels

        else:
            # self.M = np.mat(np.random.rand(num_points,20))
            data = np.mat(scipy.sparse.random(num_points,20, density = 0.1).A)
            labels = np.zeros(num_points)
            synth_pca = PCA(n_components=2)
            pca_init = synth_pca.fit_transform(data)
            self.M = data
            self.pca_init = pca_init
            self.labels = labels


        print("M: {}".format(self.M.shape))

        self.k = k #intended number of clusters in latent space

        # RANDOM INITIALIZATION
        # self.P = 10*np.mat(np.random.rand(self.m,self.d))

        # PCA INITIALIZATION
        pca = PCA(n_components=self.d)
        pca.fit_transform(self.M)
        self.P = np.mat(pca.components_).T

        print("P: {}".format(self.P.shape))

        self.method = 'adam'
        # self.method = 'BFGS'
        # self.method = 'gradient ascent'

        self.lr = 0.01
        self.lambda_ = 0 #
        self.num_epochs = 50 #maximum number of epochs for optimization

    def dump(self):
        return (self.M, self.P, self.k,
                self.lambda_, self.lr,
                self.num_epochs,
                self.random_state, self.method,
                self.pca_init, self.labels)
