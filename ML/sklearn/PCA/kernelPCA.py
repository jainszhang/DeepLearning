#coding=utf-8


'''核PCA通过核方法实现非线性降维，它具有许多应用，包括去噪, 压缩和结构化预测。KernelPCA 支持 transform 和 inverse_transform 。
'''

'''例子：
展示了线性可分的例子'''
def kernelPCA():

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.decomposition import PCA,KernelPCA
    from sklearn.datasets import make_circles
    np.random.seed(0)
    x,y = make_circles(n_samples=400,factor=.3,noise=0.05)
    kpca = KernelPCA(kernel="rbf",fit_inverse_transform=True,gamma=10)
    x_kpca = kpca.fit_transform(x)#kernel PCA 得到的值
    x_back = x_kpca.invers_transform(x_kpca)#映射后数据逆映射的到原始数据维度上
    pca = PCA()
    x_pca = pca.fit_transform(x)#PCA得到的值



    #画图
    plt.figure()
    plt.subplot(2,2,1,aspect='equal')
    plt.title("Original space")

    reds = y == 0
    blues = y == 1

    plt.scatter(x[reds,0],x[reds,1],c="red",s = 20,edgecolors='k')
    plt.scatter(x[blues,0],x[reds,1],c="red",s = 20,edgecolors='k')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    x1, x2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))#从两个或多个坐标向量返回坐标矩阵。
    x_grid = np.array([np.ravel(x1), np.ravel(x2)]).T#作出坐标矩阵

    #主成分映射
    z_grid = kpca.transform(x_grid)[:,0].reshape(x1.shape)
    plt.contour(x1,x2,z_grid,color='grey',linewidth=1,origin='lower')

    plt.subplot(2,2,2,aspect = 'equal')
    plt.scatter(x_pca[reds, 0], x_pca[reds, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(x_pca[blues, 0], x_pca[blues, 1], c="blue",
                s=20, edgecolor='k')
    plt.title("Projection by PCA")
    plt.xlabel("1st principal component")
    plt.ylabel("2nd component")

    plt.subplot(2, 2, 3, aspect='equal')
    plt.scatter(x_kpca[reds, 0], x_kpca[reds, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(x_kpca[blues, 0], x_kpca[blues, 1], c="blue",
                s=20, edgecolor='k')
    plt.title("Projection by KPCA")
    plt.xlabel("1st principal component in space induced by $\phi$")
    plt.ylabel("2nd component")

    plt.subplot(2, 2, 4, aspect='equal')
    plt.scatter(x_back[reds, 0], x_back[reds, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(x_back[blues, 0], x_back[blues, 1], c="blue",
                s=20, edgecolor='k')
    plt.title("Original space after inverse transform")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

    plt.show()
kernelPCA()