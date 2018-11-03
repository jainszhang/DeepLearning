# -*- coding=utf-8 -*-

# from scatwave.scattering import Scattering
import torch

import tensorflow as tf
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets

import os
from PIL import Image
import numpy as np
import face_recognition#the recognize face package

import sklearn
from sklearn.decomposition import PCA


def cut_celeba_face():
    list = os.listdir('/Users/jains/datasets/img_align_celeba/')

    WIDTH = 128#the aim dim
    HIGHT = 128

    for i in range(0, 100):
        imgName = os.path.join('/Users/jains/datasets/img_align_celeba/', os.path.basename(list[i]))
        fileName = os.path.basename(list[i])
        # print imgName
        if (os.path.splitext(imgName)[1] != '.jpg'): continue

        image = face_recognition.load_image_file(imgName)#recongnition the face

        face_locations = face_recognition.face_locations(image)#return face locations in the image

        for face_location in face_locations:

            top, right, bottom, left = face_location#detail of locations

            x = (top + bottom) / 2#the center of faces
            y = (right + left) / 2

            #compute details of face loactions
            top = x - HIGHT / 2
            bottom = x + HIGHT / 2
            left = y - WIDTH / 2
            right = y + WIDTH / 2

            #let faces' width equal higth
            if (top < 0) or (bottom > image.shape[0]) or (left < 0) or (right > image.shape[1]):
                top, right, bottom, left = face_location
                width = right - left
                height = bottom - top
                if (width > height):
                    right -= (width - height)
                elif (height > width):
                    bottom -= (height - width)

            #cut face from original images
            face_image = image[top:bottom, left:right]

            #translate into PIL data
            pil_image = Image.fromarray(face_image)
            pil_image = pil_image.resize((128,128))#resize to the fixed dim images
            pil_image.save('/Users/jains/datasets/face/%s'%fileName)#save the faces
        if i%10 == 0:
            print('the number of images is {}'.format(i))

def resize_face():
    source_files = './datasets/diracs/1024/'
    filename_list = os.listdir(source_files)

    for i in range(len(filename_list)):
        filename = filename_list[i]
        imgName = os.path.join(source_files, os.path.basename(filename_list[i]))
        if (os.path.splitext(imgName)[1] != '.jpg'):continue
        img = Image.open(imgName).resize((128,128))
        img.save('./datasets/celebA/256/%s'%filename)



def scat_data(data_dir,outdata_dir,M,N,J):
    filename_list = os.listdir(data_dir)#read the directory files's name
    scat = Scattering(M=M, N=N, J=J).cuda()  # scattering transform
    for i in range(0,len(filename_list)):
        imgName = os.path.join(data_dir, os.path.basename(filename_list[i]))
        if (os.path.splitext(imgName)[1] != '.jpg'): continue
        # img = np.array(Image.open(imgName)) / 255.
        img = np.array(Image.open(imgName))
        img = img.reshape(1,M,N,3)
        img_data = img.transpose(0, 3, 1, 2).astype(np.float32)  # change dim
        img_data = torch.from_numpy(img_data).cuda()
        out_data = np.array(scat(img_data))
        # print out_data.shape
        str1 = filename_list[i].split('.')
        np.save(outdata_dir + str1[0] + '.npy',out_data)
        if i%100 ==0:
            print ("step is %d",i)


def PCA_example():
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris

    # PCA类把主成分的数量作为超参数，和其他估计器一样，PCA也用fit_transform()返回降维的数据矩阵

    data = load_iris()

    y = data.target
    X = data.data
    print X.shape
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X)

    # 把图形画出来
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_X)):
        if y[i] == 0:
            red_x.append(reduced_X[i][0])
            red_y.append(reduced_X[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_X[i][0])
            blue_y.append(reduced_X[i][1])
        else:
            green_x.append(reduced_X[i][0])
            green_y.append(reduced_X[i][1])

    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()


def pca_data(data_dir):
    filename_list = os.listdir(data_dir)  # read the directory files's name
    alldata = []
    for i in range (len(filename_list)):
        tmp = np.load(data_dir+filename_list[i])
        alldata.append(tmp)
        print ("append step %d"%i)
    alldata = np.array(alldata)
    alldata = alldata.reshape(len(alldata),-1)

    pca = PCA(n_components=512,copy=True, whiten=True, svd_solver='randomized')
    X = pca.fit_transform(alldata)
    # np.save('10000.npy',X)

    for i in range(len(filename_list)):
        np.save('./datasets/celebA_128/ScatJ4_projected512_1norm/'+ os.path.splitext(filename_list[i])[0] + '.npy',X[i])
        print ("step is %d"%i)












    # pca = PCA( copy=True,whiten=True,svd_solver='randomized')
    #
    #
    # tmp = np.load(data_dir+'000002.npy')
    # alldata.append(tmp)
    # data = np.load(data_dir+'000001.npy')
    # alldata.append(data)
    # tmp = np.load(data_dir+'000005.npy')
    # alldata.append(tmp)
    # tmp = np.load(data_dir + '000006.npy')
    # alldata.append(tmp)
    # data = np.array(alldata)
    # data = data.reshape(len(alldata),-1)
    # # print data.shape
    # new_data = pca.fit_transform(data)
    # print new_data.shape

inputdata_dir = './datasets/celebA/256/'
outputdata_dir = './datasets/celebA/256_ScatJ4/'
M = 128;N = 128;J = 4


# scat_data(inputdata_dir,outputdata_dir,M,N,J)

# pca_data('./datasets/celebA_128/128_ScatJ4/')




def pca_mnist():

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    X = mnist["data"]
    print type(X),X.shape
    # 使用np.array_split（）方法的IPCA
    from sklearn.decomposition import IncrementalPCA
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X, n_batches):
        inc_pca.partial_fit(X_batch)
    inc_pca.partial_fit(X)

    X_mnist_reduced = inc_pca.transform(X)
    print X_mnist_reduced.shape

# pca_mnist()

# data = np.load('epoch_1.pth')
# print type(data)
# resize_face()



cut_celeba_face()


# img_data = np.array(img_data)
# img_data = img_data.transpose(0,3,1,2).astype(np.float32)#change dim
# print img_data.shape


# trainloader = torch.utils.data.DataLoader(img_data,batch_size = 1,shuffle=True,num_workers=2)#train loader
# for i,iter_data in enumerate(trainloader):
#     iter_data = iter_data.cuda()
#     scat = Scattering(M=128, N=128, J=2).cuda()  # scattering transform
#     out = scat(iter_data)
#
#     scat_data = np.array(out)
#     np.save('./datasets/diracs/1024_ScatJ4/'+filename_list[i]+'.npy')

# dataiter = iter(trainloader)
# data = next(dataiter)
# data = data.cuda()
# scat = Scattering(M=128, N=128, J=2).cuda()#scattering transform
# out = scat(data)
# out_data = np.array(out)
# # np.save("out_data.npy",out_data)
# print out.size()





#
#
# ''' 读取MNIST数据方法一'''
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)
# '''1)获得数据集的个数'''
# train_nums = mnist.train.num_examples
# validation_nums = mnist.validation.num_examples
# test_nums = mnist.test.num_examples
# '''2)获得数据值'''
# train_data = mnist.train.images   #所有训练数据
# val_data = mnist.validation.images  #(5000,784)
# test_data = mnist.test.images       #(10000,784)
# """获取标签"""
# train_labels = mnist.train.labels     #(55000,10)
# val_labels = mnist.validation.labels  #(5000,10)
# test_labels = mnist.test.labels       #(10000,10)
#
# data = torchvision.datasets.ImageFolder(root="./../datasets/cut_face")
#
# # data = torchvision.datasets.ImageFolder(root="./../datasets/face", transforms.ToTensor())
#
#
# data_dir = './../datasets/cut_face/'
# # trans data
# image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['face/']}
# # load data
# # data_loaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}
#
# # data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# # class_names = image_datasets['train'].classes
# # print(data_sizes, class_names)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
# trainloader = torch.utils.data.DataLoader(data,batch_size = 4,shuffle=True,num_workers=2)
# # testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
# testloader = torch.utils.data.DataLoader(test_data,batch_size = 4,shuffle = False,num_workers = 2)

# plt.figure()
# for i in range(100):
#     im = train_data[i].reshape(28,28)
#     # im = batch_xs[i].reshape(28,28)
#     plt.imshow(im,'gray')
#     plt.pause(0.0000001)
# plt.show()
# im = train_data[0].reshape(28,28)
# print (type(im))
# im = torch.from_numpy(im)
# im = im.cuda()
# print(im)
# scat = Scattering(M=28, N=28, J=2).cuda()
# # out = scat(im)




# dataiter = iter(trainloader)
# data = next(dataiter)
#
# data = data.view(-1,1,28,28)
# data = data.cuda()
# scat = Scattering(M=28, N=28, J=2).cuda()
# out = scat(data)
# print out.size()





# for i, data in enumerate(trainloader, 0):
#       inputs = data
#       inputs = inputs.view(4,1,28,28)
      # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())  # 包装数据
