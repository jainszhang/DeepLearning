from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt

x=np.array([[3,3],
            [4,3],
            [1,1]])
y=np.array([1,1,-1])


plt.scatter(x[:,0],x[:,1])

def sig(x):
    return np.sign(x)

rate = 1
w = np.zeros((2,1))
b = np.zeros((1))
Y1 = -w[0] * x / w[1]
# plt.plot(x,Y1,c='b')

print(len(x))

for i in range(100):
    for j in range(3):
        x1 = x[j].reshape((2,1))
        y1 = y[j].reshape((1,1))

        if y1*(np.dot(w.transpose(),x1) + b)[0] <= 0 :
            w = w + rate * y1*x1
            b = b + rate*y1

            #loss = np.dot(y.transpose(),np.dot(x,w)+b)


print(w)
print(b)
Y2 = -w[0] * x / w[1]+b[0][0]
plt.plot(x,Y2,c='r')

plt.show()