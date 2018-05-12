import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#kd树存储与搜索
class Node:
    def __init__(self, data, lchild = None, rchild = None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild

    def create(self, dataSet, depth):  # 创建kd树，返回根结点
        if (len(dataSet) > 0):
            m, n = np.shape(dataSet)  # 求出样本行，列
            midIndex = m / 2  # 中间数的索引位置
            axis = depth % n  # 判断以哪个轴划分数据，对应书中算法3.2（2）公式j()
            sortedDataSet = self.sort(dataSet, axis)  # 进行排序
            node = Node(sortedDataSet[midIndex])  # 将节点数据域设置为中位数，具体参考下书本
            # print sortedDataSet[midIndex]
            leftDataSet = sortedDataSet[: midIndex]  # 将中位数的左边创建2个副本
            rightDataSet = sortedDataSet[midIndex + 1:]
            print(leftDataSet)
            print(rightDataSet)
            node.lchild = self.create(leftDataSet, depth + 1)  # 将中位数左边样本传入来递归创建树
            node.rchild = self.create(rightDataSet, depth + 1)
            return node
        else:
            return None

    def sort(self, dataSet, axis):  # 采用冒泡排序，利用aixs作为轴进行划分
        sortDataSet = dataSet[:]  # 由于不能破坏原样本，此处建立一个副本
        m, n = np.shape(sortDataSet)
        for i in range(m):
            for j in range(0, m - i - 1):
                if (sortDataSet[j][axis] > sortDataSet[j + 1][axis]):
                    temp = sortDataSet[j]
                    sortDataSet[j] = sortDataSet[j + 1]
                    sortDataSet[j + 1] = temp
        print(sortDataSet)
        return sortDataSet

    def preOrder(self, node):
        if node != None:
            print("tttt->%s" % node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

    #kd树搜索
    def search(self, tree, x):  # 搜索
        self.nearestPoint = None  # 保存最近的点
        self.nearestValue = 0  # 保存最近的值

        def travel(node, depth=0):  # 递归搜索
            if node != None:  # 递归终止条件
                n = len(x)  # 特征数
                axis = depth % n  # 计算轴
                if x[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
                    travel(node.lchild, depth + 1)
                else:
                    travel(node.rchild, depth + 1)

                    # 以下是递归完毕，对应算法3.3(3)
                distNodeAndX = self.dist(x, node.data)  # 目标和节点的距离判断
                if (self.nearestPoint == None):  # 确定当前点，更新最近的点和最近的值，对应算法3.3(3)(a)
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                elif (self.nearestValue > distNodeAndX):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX

                print(node.data, depth, self.nearestValue, node.data[axis], x[axis])
                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                    if x[axis] < node.data[axis]:
                        travel(node.rchild, depth + 1)
                    else:
                        travel(node.lchild, depth + 1)

        travel(tree)
        return self.nearestPoint


    def dist(self, x1, x2):  # 欧式距离的计算
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5







n_samples = 5000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-1, -1), (5, 5)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])
X_train, X_test, y_train, y_test, sw_train, sw_test = \
    train_test_split(X, y, sample_weight, test_size=0.1, random_state=42)

# print(X_train.shape)

def distance(point,x):
    dist = np.sqrt(np.sum(np.square(np.subtract(x,point)),axis=1))
    return dist


k = 500
sum1 = 0

def result(dist,k):
    index = dist.argsort()#获取排序后的数组下标
    index = index[:k]
    out = y_train[index].tolist()
    return out.count(0) < k - out.count(0)


for i in range(len(X_test)):
    dist = distance(X_test[i], X_train)
    sum1 = sum1 + np.equal(y_test[i],result(dist,k))#y_train[dist.index(min(dist))])

print(np.float(sum1/len(X_test)))

plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.scatter(X_test[:,0],X_test[:,1],c='b')
plt.show()