#coding=utf-8

import  os
'''例子1

，使用PCA和LDA压缩IRIS数据集'''
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





'''例子2：
使用PCA和FA对比，寻找最优的参数'''
'''1)
FactorAnalysis：因子分析用于模型选择和协方差估计，给出了低纬度的数据加入不同的噪音进行分析
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





'''例子3：
使用随机SVD的PCA用于人脸识别
1) classification_report:分类报告，显示主要的分类指标，返回每个类标签的精确、召回率及F1值
2）fetch_lfw_people：加载有标签的原生人脸图
3）confusion_matrix：混淆矩阵是机器学习中总结分类模型预测结果的情形分析表，以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总

'''

'''PCA使用随机SVD：通过丢弃具有较低奇异值的奇异向量成分，将数据降维到低维空间并保留大部分方差是非常有意义的。'''
def pca_face_recognition():

    from time import time
    import logging
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV#网格搜索最优参数
    from sklearn.datasets import fetch_lfw_people#加载有标签的原生人脸图
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    logging.basicConfig(level=logging.INFO,format='%(asctime)%s %(message)s')
    #原始图像为250*250
    lfw_people = fetch_lfw_people(data_home='/Users/jains/datasets/sklearn-datasets/',min_faces_per_person=70,resize=0.4)
    n_samples,h,w = lfw_people.images.shape#1288个样本，h=50，w=37

    x = lfw_people.data#获取数据   37*50=1850
    n_features = x.shape[1]#获取数据特征维数

    y = lfw_people.target#1288个对应的标签
    target_names = lfw_people.target_names#标签对应的名字
    n_classes = target_names.shape[0]#获取类别个数--7个类

    print("total dataset size:")
    print("n_samples:%d"%n_samples)
    print("n_features:%d"%n_features)
    print("n_classes:%d"%n_classes)


    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)#分割训练集和测试集
    #训练样本为966，测试样本为322
    n_components = 150

    print("从%d个特征中提取前%d个特征"%(x_train.shape[1],n_components))
    t0 = time()
    pca = PCA(n_components=n_components,svd_solver='randomized',whiten=True).fit(x_train)#此过程白化操作
    print("PCA done in %.3f"%(time()-t0))

    eigenfaces = pca.components_.reshape((n_components,h,w))#150*50*37

    print("把输入数据映射到压缩后的基上")

    t0 = time()
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)#322*150

    print("PCA done in %.3f"%(time()-t0))

    #训练SVM分类模型
    t0 = time()
    param_grid = {'C':[1e3,5e3,1e4,5e4,1e5],'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)#寻找最合适的参数值
    clf = clf.fit(x_train_pca, y_train)
    print ("Done in %.3f"%(time()-t0))
    print ("找到的最好评估为：")
    print(clf.best_estimator_)

    #测试集上评估分类情况
    print("测试集评估分类质量")
    t0 = time()
    y_pred = clf.predict(x_test_pca)
    print ("done in %.3f"%(time() -  t0))
    print(classification_report(y_test, y_pred, target_names=target_names))#用于分析预测结果，预测报告，F1，召回率等等
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))#混淆矩阵分析预测结果



    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the result of the prediction on a portion of the test set

    def title(y_pred, y_test, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

    prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

    plot_gallery(x_test, prediction_titles, h, w)

    # plot the gallery of the most significative eigenfaces

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()
# pca_face_recognition()




'''例4：压缩人脸数据集
使用的是随机SVD的PCA压缩的
'''

def faces_decomposition():
    import logging
    from numpy.random import RandomState#随机数生成器种子，从高斯分布或者其他等分布产生
    import matplotlib.pyplot as plt
    from time import time
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.cluster import MiniBatchKMeans
    from sklearn import decomposition

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    n_row,n_col = 2,3
    n_components = n_row * n_col
    image_shape = (64,64)
    rng = RandomState(0)


    #加载数据集
    dataset = fetch_olivetti_faces(shuffle=True,random_state=rng)
    faces = dataset.data

    n_samples,n_features = faces.shape

    faces_centered = faces - faces.mean(axis = 0)

    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples,-1)

    print("dataset consits of %d faces"%n_samples)#样本个数

    def plot_gallery(title, images, n_col=n_col, n_row=n_row):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    estimators = [
        ('Eigenfaces - PCA using randomized SVD',
         decomposition.PCA(n_components=n_components, svd_solver='randomized',
                           whiten=True),
         True),

        ('Non-negative components - NMF',
         decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
         False),

        ('Independent components - FastICA',
         decomposition.FastICA(n_components=n_components, whiten=True),
         True),

        ('Sparse comp. - MiniBatchSparsePCA',
         decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                                          n_iter=100, batch_size=3,
                                          random_state=rng),
         True),

        ('MiniBatchDictionaryLearning',
         decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                                   n_iter=50, batch_size=3,
                                                   random_state=rng),
         True),

        ('Cluster centers - MiniBatchKMeans',
         MiniBatchKMeans(n_clusters=n_components, tol=1e-3, batch_size=20,
                         max_iter=50, random_state=rng),
         True),

        ('Factor Analysis components - FA',
         decomposition.FactorAnalysis(n_components=n_components, max_iter=2),
         True),
    ]

    # #############################################################################
    # Plot a sample of the input data

    plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

    # #############################################################################
    # Do the estimation and plot it

    for name, estimator, center in estimators:
        print("Extracting the top %d %s..." % (n_components, name))
        t0 = time()
        data = faces
        if center:
            data = faces_centered
        estimator.fit(data)
        train_time = (time() - t0)
        print("done in %0.3fs" % train_time)
        if hasattr(estimator, 'cluster_centers_'):
            components_ = estimator.cluster_centers_
        else:
            components_ = estimator.components_

        # Plot an image representing the pixelwise variance provided by the
        # estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
        # via the PCA decomposition, also provides a scalar noise_variance_
        # (the mean of pixelwise variance) that cannot be displayed as an image
        # so we skip it.
        if (hasattr(estimator, 'noise_variance_') and
                estimator.noise_variance_.ndim > 0):  # Skip the Eigenfaces case
            plot_gallery("Pixelwise variance",
                         estimator.noise_variance_.reshape(1, -1), n_col=1,
                         n_row=1)
        plot_gallery('%s - Train time %.1fs' % (name, train_time),
                     components_[:n_components])

    plt.show()
faces_decomposition()