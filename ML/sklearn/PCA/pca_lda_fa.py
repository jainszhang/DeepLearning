#coding=utf-8

import  os
'''最简单的PCA，小量的数据集'''

'''PCA参数简介
1）n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目，默认情况n_components=min(样本数，特征数)
2）whiten：判断是否进行白化，白化就是对降维后的数据的每个特征进行归一化，默认False
3）svd_solver:即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}，
randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 full则是传统意义上的SVD，使用了scipy库对应的实现。
arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现。
默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。
'''

'''LDA简介
线性判别分析是一种分类模型，它通过在k维空间选择一个投影超平面，使得不同类别在该超平面上的投影之间的距离尽可能近，同时不同类别的投影之间的距离尽可能远
n_components：需要保留的特征的个数
'''

def pca_vs_lda():
    import matplotlib.pyplot as plt

    from sklearn import  datasets
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis#线性判别分析

    iris = datasets.load_iris()
    x = iris.data#150*4
    y = iris.target#150*1

    target_names = iris.target_names#名字3*1,表示3中花名

    pca = PCA(n_components=2)#数据纬度降低为2维
    x_r = pca.fit(x)
    x_r = x_r.transform(x)#150*2,降纬后的数据x，表示转换为numpy数组

    lda = LinearDiscriminantAnalysis(n_components=2)#申请线性判别分析
    x_r2 = lda.fit(x,y).transform(x)#放入数据x和标签y

    print('explained variance ratio for each components %s'%str(pca.explained_variance_ratio_))#pca.explained_variance_ratio_表示各个纬度方差所占比例
    print('explained variance for each components %s'%str(pca.explained_variance_))#表示每个维度的方差

    plt.figure()#用来展示分类钱的数据
    colors = ['navy','turquoise','darkorange']
    lw = 2

    for color,i,target_names in zip(colors,[0,1,2],target_names):
        plt.scatter(x_r[y == i,0],x_r[y == i,1],color=color,alpha=.8,lw=lw,label=target_names)
    plt.legend(loc = 'best',shadow=False,scatterpoints=1)
    plt.title('PCA of IRIS dataset')
    # plt.show()

    plt.figure()#用来展示LDA方法压缩后的数据
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(x_r2[y == i, 0], x_r2[y == i, 1], alpha=.8, color=color,label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')

    plt.show()



'''1)FactorAnalysis：因子分析用于模型选择和协方差估计，给出了低纬度的数据加入不同的噪音进行分析
'''
def pca_vs_fa():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import linalg

    from sklearn.decomposition import PCA,FactorAnalysis
    from sklearn.covariance import ShrunkCovariance,LedoitWolf
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV

    #创建数据
    n_samples, n_features, rank = 1000, 50, 10
    sigma = 1.
    rng = np.random.RandomState(42)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    x = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

    # Adding homoscedastic noise
    x_homo = x + sigma * rng.randn(n_samples, n_features)#加入不同噪音的x

    # Adding heteroscedastic noise
    sigmas = sigma * rng.rand(n_features) + sigma / 2.
    x_hetero = x + rng.randn(n_samples, n_features) * sigmas


    #喂入数据、
    n_components = np.arange(0,n_features,5)#尝试多个保留的纬度，5个间隔  [0,5,10...]

    def compute_scores(x):
        pca = PCA(svd_solver='full')#申请模型
        fa = FactorAnalysis()

        pca_scores,fa_scores = [],[]
        for n in n_components:
            pca.n_components = n
            fa.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca,x)))#通过交叉验证估计得分
            fa_scores.append(np.mean(cross_val_score(fa,x)))
        return pca_scores,fa_scores

    '''
    1）GridSearchCV：它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，一旦数据的量级上去了，
    很难得出结果。这个时候就是需要动脑筋了。数据量比较大的时候可以使用一个快速调优的方法——坐标下降。它其实是一种贪心算法：拿当前对模型影响最大的参数调优，直到最优化；
    再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时间省力，
    grid.fit()：运行网格搜索
    grid_scores_：给出不同参数情况下的评价结果
    best_params_：描述了已取得最佳结果的参数的组合
    best_score_：成员提供优化过程期间观察到的最好的评分

    2）ShrunkCovariance：协方差估计，协方差主要是用来估计数据中不同特征之间的相互关系的一个统计量，除协方差以外，"相关系数”也可以用来估计数据中不同特征的相互关系
    3）cross_val_score：交叉验证，每一个 k 折都会遵循下面的过程：
                        将 k-1 份训练集子集作为 training data （训练集）训练模型，
                        将剩余的 1 份训练集子集作为验证集用于模型验证
    '''
    def shrunk_cov_score(x):
        shrinkages = np.logspace(-2,0,30)#-2~0之间取30个值
        cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})#网格搜索最优参数
        return np.mean(cross_val_score(cv.fit(x).best_estimator_, x))

    def lw_score(X):
        return np.mean(cross_val_score(LedoitWolf(), X))

    for X, title in [(x_homo, 'Homoscedastic Noise'),
                     (x_hetero, 'Heteroscedastic Noise')]:
        pca_scores, fa_scores = compute_scores(X)#两种方式交叉验证的得分
        n_components_pca = n_components[np.argmax(pca_scores)]#最大得分的pca压缩特征数
        n_components_fa = n_components[np.argmax(fa_scores)]

        pca = PCA(svd_solver='full', n_components='mle')
        pca.fit(X)
        n_components_pca_mle = pca.n_components_

        print("best n_components by PCA CV = %d" % n_components_pca)
        print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
        print("best n_components by PCA MLE = %d" % n_components_pca_mle)

        plt.figure()
        plt.plot(n_components, pca_scores, 'b', label='PCA scores')
        plt.plot(n_components, fa_scores, 'r', label='FA scores')
        plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
        plt.axvline(n_components_pca, color='b',
                    label='PCA CV: %d' % n_components_pca, linestyle='--')
        plt.axvline(n_components_fa, color='r',
                    label='FactorAnalysis CV: %d' % n_components_fa,
                    linestyle='--')
        plt.axvline(n_components_pca_mle, color='k',
                    label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

        # compare with other covariance estimators
        plt.axhline(shrunk_cov_score(X), color='violet',
                    label='Shrunk Covariance MLE', linestyle='-.')
        plt.axhline(lw_score(X), color='orange',
                    label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

        plt.xlabel('nb of components')
        plt.ylabel('CV scores')
        plt.legend(loc='lower right')
        plt.title(title)

    plt.show()

# pca_vs_fa()

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(X.shape,y.shape)
print(cross_val_score(lasso, X, y))
