# -*- coding: utf-8 -*-
import pickle as p
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

def load_MNIST(file_dir):
    return input_data.read_data_sets(file_dir,one_hot = True)


def load_CIFAR_batch(filename):
    """ 载入cifar数据集的一个batch """
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)#.astype("float")
        Y = np.array(Y)
    return X, Y

def load_CIFAR10(file_dir):
    """ 载入cifar全部数据 """
    xs = []
    ys = []
    Xtr = []
    Ytr = []
    for b in range(1,6):
        data_file = os.path.join(file_dir, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(data_file)

        xs.append(X) #将所有batch整合起来
        ys.append(Y)
        Xtr = np.concatenate(xs) #使变成行向量,最终Xtr的尺寸为(50000,32,32,3)
        Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR_batch(os.path.join(file_dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte



# cifar10_dir = 'data_set/cifar-10-batches-py/'
# X_train, Y_train, X_test, Y_test  = load_CIFAR10(cifar10_dir)
#
# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = p.load(fo, encoding='bytes')
#     return dict
# classes = unpickle(cifar10_dir+"batches.meta")
# print(classes)