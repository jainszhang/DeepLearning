from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt

x, y = make_classification(
    n_samples=500, n_features=2,
    n_redundant=0, n_informative=2,
    random_state=1000, n_clusters_per_class=1,
    scale=1000)

x = preprocessing.scale(x)



def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def gradient_Ascent(x,y):
    y_ = y
    y = np.reshape(y,(y.shape[0],1))
    weights = np.array([[0.1],
                        [0.9]])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x[:, 0], x[:, 1], c=y_)
    y1 = -weights[0] * x / weights[1]
    plt.plot(x,y1)
    plt.ion()
    plt.show()

    for i in range(500):
        s = sigmoid(np.dot(x,weights))
        tmp = y - s
        weights = weights + 0.0001 * np.dot(x.transpose(), tmp)

    return weights

w = gradient_Ascent(x,y)
print(w)
y2 = -w[0] * x / w[1]
plt.plot(x,y2,c = 'green')
plt.pause(90)




