import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

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

index = 8
dist = distance(X_test[index],X_train)
dist = dist.tolist()
print(dist.index(min(dist)))
print(y_train[dist.index(min(dist))])
print(y_test[index])
# plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
# plt.scatter(X_test[:,0],X_test[:,1],c='b')
# plt.show()