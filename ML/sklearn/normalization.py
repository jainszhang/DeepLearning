# from sklearn import preprocessing #标准化数据模块
# import numpy as np
# import matplotlib.pyplot as plt
#

#建立Array
# a = np.array([[10, 2.7, 3.6],
#               [-100, 5, -2],
#               [120, 20, 40]], dtype=np.float64)
#
# print(a)
# print(preprocessing.scale(a))








#例子2
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt



x, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2,
    random_state=22, n_clusters_per_class=1,
    scale=100)

plt.scatter(x[:,0],x[:,1],c = y)
plt.show()
#是否归一化会对结果造成很大影响
x = preprocessing.minmax_scale(x,feature_range=(0,1))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3)
clf = SVC()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))