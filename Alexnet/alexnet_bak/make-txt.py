import os
import numpy as np
import math
zeroclass = []
label_zeroclass = []
oneclass = []
label_oneclass = []

file_dir = 'id_data/'#原始图片地址
f_1 = open('train_data/train.txt', 'a+')#存储训练数据的文件
f_2 = open('train_data/val.txt', 'a+')#存储测试数据的文件

for file in os.listdir(file_dir + 'id'):
    zeroclass.append(file_dir + 'id' + '/' + file)#添加图片完整路径
    label_zeroclass.append('0')#添加对应标签
for file in os.listdir(file_dir + 'other'):
    oneclass.append(file_dir + 'other' + '/' + file)
    label_oneclass.append('1')#添加标签

# s2 对生成图片路径和标签list打乱处理（img和label）
image_list = np.hstack((zeroclass, oneclass))
label_list = np.hstack((label_zeroclass, label_oneclass))
# shuffle打乱
temp = np.array([image_list, label_list])
temp = temp.transpose()
np.random.shuffle(temp)
# 将所有的img和lab转换成list
all_image_list = list(temp[:, 0])
all_label_list = list(temp[:, 1])
# 将所得List分为2部分，一部分train,一部分val，ratio是验证集比例
n_sample = len(all_label_list)
n_val = int(math.ceil(n_sample * 0.2))  # 验证样本数
n_train = n_sample - n_val  # 训练样本数

tra_images = all_image_list[0:n_train]
tra_labels = all_label_list[0:n_train]
#tra_labels = [int(float(i)) for i in tra_labels]
val_images = all_image_list[n_train:]
val_labels = all_label_list[n_train:]
#val_labels = [int(float(i)) for i in val_labels]
for i in range(len(tra_images)):
    # print(tra_images[i])
    # print(tra_labels[i])
    f_1.write(tra_images[i] + ' ' + tra_labels[i] + '\n')
for i in range(len(val_images)):
    f_2.write(val_images[i] + ' ' + val_labels[i] + '\n')

f_1.close()
f_2.close()



