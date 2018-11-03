# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""
# coding=utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from EmbeddingsImagesDataset import EmbeddingsImagesDataset

dir_datasets = os.path.expanduser('./datasets')
dataset = 'diracs'
dataset_attribute = '1024'
embedding_attribute = 'ScatJ4'

dir_x_train = os.path.join(dir_datasets, dataset, '{0}'.format(dataset_attribute))
dir_z_train = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(dataset_attribute, embedding_attribute))

# print(dir_x_train)

dataset = EmbeddingsImagesDataset(dir_z_train, dir_x_train, nb_channels=3)#表示嵌入过程，训练的z和x的嵌入
# fixed_dataloader = DataLoader(dataset, batch_size=256)#加载数据集
fixed_dataloader = DataLoader(dataset, batch_size=256)
fixed_batch = next(iter(fixed_dataloader))

x = fixed_batch['x'].numpy()
z = fixed_batch['z'].numpy()

min_distance = np.inf#表示无穷大
i_tilde = 0
j_tilde = 0

distances = list()
for i in range(256):
    for j in range(256):
        if i < j:
            temp = (z[i] - z[j]) ** 2
            temp = np.sum(temp)
            temp = np.sqrt(temp)

            if temp < min_distance:
                min_distance = temp
                i_tilde = i
                j_tilde = j

            distances.append(temp)

distances = np.array(distances)
print('Most similar indexes:', i_tilde, j_tilde)

print('Min distances:', distances.min())

plt.hist(distances)
plt.show()
