#coding=utf-8

'''
PCA对于大数据有限制，对大量数据使用PCA时必须批处理适应内存， IncrementalPCA 对象使用不同的处理形式使之允许部分计算， 这一形式几乎和 PCA 以小型批处理方式处理数据的方法完全匹配
IncrementalPCA 可以通过以下方式实现核外（out-of-core）主成分分析：
    >使用 partial_fit 方法从本地硬盘或网络数据库中以此获取数据块。
    >通过 numpy.memmap 在一个 memory mapped file 上使用 fit 方法。
    IncrementalPCA 仅存储成分和噪声方差的估计值，并按顺序递增地更新解释方差比（explained_variance_ratio_）。

可以理解为快速的PCA算法，可以处理大型数据
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA,IncrementalPCA

iris = load_iris()
x = iris.data
y = iris.target

n_components = 2
ipca = IncrementalPCA(n_components=n_components,batch_size=10)
x_ipca = ipca.fit_transform(x)#pca后直接转换为数组

pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x)
print x_ipca.shape,x_pca.shape

colors = ['navy', 'turquoise', 'darkorange']
for X_transformed, title in [(x_ipca, "Incremental PCA"), (x_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                    color=color, lw=2, label=target_name)

    if "Incremental" in title:
        err = np.abs(np.abs(x_pca) - np.abs(x_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error "
                  "%.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()