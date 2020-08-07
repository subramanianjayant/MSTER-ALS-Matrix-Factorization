from sklearn import manifold, decomposition
from sklearn import metrics
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = np.array(pd.read_csv('radar.csv', header = None))
labels = np.array(pd.read_csv('radar_labels.csv', header = None))

metric_2d = manifold.MDS(n_components = 2, metric = True)
nonmetric_2d = manifold.MDS(n_components = 2, metric = False)
metric_131d = manifold.MDS(n_components = 351, metric = True)
nonmetric_131d = manifold.MDS(n_components = 351, metric = False)

m2d = metric_2d.fit_transform(data)
nm2d = nonmetric_2d.fit_transform(data)
m131d = metric_131d.fit_transform(data)
nm131d = nonmetric_131d.fit_transform(data)

pca1 = decomposition.PCA(n_components = 2)
pca2 = decomposition.PCA(n_components = 2)

m131d_pca = pca1.fit_transform(m131d)
nm131d_pca = pca2.fit_transform(nm131d)

mtsne = manifold.TSNE(n_components=2).fit_transform(m131d)
nmtsne = manifold.TSNE(n_components=2).fit_transform(nm131d)

recon_data_nm131d = metrics.pairwise_distances(m131d)
print("\n\nSUBTRACTION")
print(data - recon_data_nm131d)
# np.savetxt('RADAR_mMDS_351d.csv', m131d, delimiter=',')

def plot_fig(n, title, arr, labels):
    _dict = {'B':[], 'G':[]}
    for i,row in enumerate(arr):
        _dict[labels[i,1]].append(row)
    plt.figure(n)
    for i in _dict.keys():
        plt.scatter(np.array(_dict[i])[:,0], np.array(_dict[i])[:,1], alpha=0.6)
    plt.legend(list(_dict.keys()))
    plt.title(title)

plot_fig(0, 'protein dataset metric MDS d=2',m2d, labels)
# plot_fig(1, 'protein dataset nonmetric MDS d=2',nm2d, labels)
plot_fig(2, 'protein dataset metric MDS+PCA d=131',m131d_pca, labels)
# plot_fig(3, 'protein dataset nonmetric MDS+PCA d=131',nm131d_pca, labels)
plot_fig(4, 'protein dataset metricMDS+TSNE d=2',mtsne, labels)
# plot_fig(5, 'protein dataset nonmetric MDS+TSNE d=2',nmtsne, labels)

plt.show()
