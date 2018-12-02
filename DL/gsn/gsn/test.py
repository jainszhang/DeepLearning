# -*- coding=utf-8 -*-


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
import scipy.io as sio
import sklearn
from sklearn.decomposition import PCA

def resize_face():
    source_files = './datasets/diracs/1024/'
    filename_list = os.listdir(source_files)

    for i in range(len(filename_list)):
        filename = filename_list[i]
        imgName = os.path.join(source_files, os.path.basename(filename_list[i]))
        if (os.path.splitext(imgName)[1] != '.jpg'):continue
        img = Image.open(imgName).resize((128,128))
        img.save('./datasets/celebA/256/%s'%filename)
    return
def cut_celeba_face():
    list = os.listdir('./../datasets/img_align_celeba/')

    WIDTH = 128#the aim dim
    HIGHT = 128

    for i in range(0, len(list)):
        imgName = os.path.join('./../datasets/img_align_celeba/', os.path.basename(list[i]))
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
            pil_image.save('./../datasets/face/%s'%fileName)#save the faces
        if i%100 == 0:
            print('the number of images is {}'.format(i))
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
def pca_mnist():

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    X = mnist["data"]
    print type(X), X.shape
    # 使用np.array_split（）方法的IPCA
    from sklearn.decomposition import IncrementalPCA
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X, n_batches):
        inc_pca.partial_fit(X_batch)
    inc_pca.partial_fit(X)

    X_mnist_reduced = inc_pca.transform(X)
    print X_mnist_reduced.shape
# def pca_data(data_dir,out_dir):
#     filename_list = os.listdir(data_dir)  # read the directory files's name
#     number = len(filename_list)
#     alldata = []
#     for i in range(len(filename_list)):
#         tmp = np.load(data_dir + filename_list[i])
#         alldata.append(tmp)
#     alldata = np.array(alldata)
#     alldata = alldata.reshape(len(alldata), -1)  # 1024*80064
#     pca = PCA(n_components=512, copy=True, whiten=True, svd_solver='randomized')
#     X = pca.fit_transform(alldata)  # 1024*512
#     # x_or = pca.inverse_transform(X)#逆转换为原始数据
#     for i in range(number):
#         str1 = filename_list[i].split('.')
#         np.save(out_dir + str1[0] + '.npy', X[i])
def pca_data(data_dir,out_dir):
    filename_list = os.listdir(data_dir)  # read the directory files's name
    number = len(filename_list)
    alldata = []
    for i in range(len(filename_list)):
        tmp = np.load(data_dir + filename_list[i])
        alldata.append(tmp)
        if (i+1)%16384 == 0:
            alldata = np.array(alldata)
            alldata = alldata.reshape(len(alldata), -1)  # 16384*80064
            pca = PCA(n_components=512, copy=True, whiten=True, svd_solver='randomized')
            X = pca.fit_transform(alldata)  # 1024*512
            # x_or = pca.inverse_transform(X)#逆转换为原始数据
            for j in range(len(alldata)):
                str1 = filename_list[j + (i/16384)*16384].split('.')
                np.save(out_dir + str1[0] + '.npy', X[j])
            alldata = []

def make_norm_data(data_dir):
    # filename_list = os.listdir(data_dir)
    for i in range(1,8):
        tmp = np.load('/home/jains/datasets/celebA/' + str(i) +'.npy')
        for j in range(len(tmp)):
            np.save('/home/jains/datasets/gsn/celebA_128/65536_ScatJ4_projected512_1norm/' + str(10000 * (i - 1) + j) + '.npy', tmp[j])
            print ("j is %d,i is %d"%(j,i))








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
def choose_img(src_dir,out_dir,num=1):
    filename_list = os.listdir(src_dir)  # read the directory files's name
    for i in range(70001,86385):
        imgName = os.path.join(src_dir, os.path.basename(filename_list[i]))
        if (os.path.splitext(imgName)[1] != '.jpg'): continue
        img = Image.open(imgName)

        img.save(out_dir + filename_list[i])
        print ("step is %d k",(i-70000)/1000)
    return 0

def scat_data(data_dir,outdata_dir,M,N,J):
    from scatwave.scattering import Scattering
    filename_list = os.listdir(data_dir)#read the directory files's name
    number = len(filename_list)
    scat = Scattering(M=M, N=N, J=J).cuda()  # scattering transform
    for i in range(0,number):
        imgName = os.path.join(data_dir, os.path.basename(filename_list[i]))
        if (os.path.splitext(imgName)[1] != '.jpg'): continue
        img = np.array(Image.open(imgName)) / 127.5 - 1#与x保持一致，归一化到[-1,1]之间
        # img = np.array(Image.open(imgName))
        # print (np.max(img),np.min(img))
        img = img.reshape(1,M,N,3)#1*128*128*3
        img_data = img.transpose(0, 3, 1, 2).astype(np.float32)  # 1*3*128*128
        img_data = torch.from_numpy(img_data).cuda()
        out_data = np.array(scat(img_data))#1*3*417*8*8
        # print (np.max(out_data),np.min(out_data))
        # print out_data.shape
        str1 = filename_list[i].split('.')
        # print (outdata_dir + str1[0] + '.npy')
        np.save(outdata_dir + str1[0] + '.npy',out_data)
        if i%100 ==0:
            print ("step is %d",i)
    return 0

train_data_dir = '/home/jains/datasets/gsndatasets/celebA_128/65536'
train_scat_dir = '/home/jains/datasets/gsndatasets/celebA_128/65536_scat/'
train_norm_dir = '/home/jains/datasets/gsndatasets/celebA_128/65536_ScatJ4_projected512_1norm/'

test_data_dir = '/home/jains/datasets/gsndatasets/celebA_128/2048_after_65536'
test_scat_dir = '/home/jains/datasets/gsndatasets/celebA_128/2048_after_65536_scat/'
test_norm_dir = '/home/jains/datasets/gsndatasets/celebA_128/2048_after_65536_ScatJ4_projected512_1norm/'
# outimg_dir = '/home/jains/datasets/gsn/celebA_128/65536/'
M = 128;N = 128;J = 4
#函数调用
scat_data(train_data_dir,train_scat_dir,M,N,J)
pca_data(train_scat_dir,train_norm_dir)
scat_data(test_data_dir,test_scat_dir,M,N,J)
pca_data(test_scat_dir,test_norm_dir)
# make_norm_data(outputdata_dir)
# cut_celeba_face()
# face_dir = '/home/jains/datasets/celebA/cut_face/'
# choose_img(face_dir,test_data_dir,1)



