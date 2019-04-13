# -*- coding=utf-8 -*-

import kymatio
import torch
# import tensorflow as tf
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets
import os
from PIL import Image
import numpy as np
# import face_recognition#the recognize face package
import scipy.io as scio
import sklearn
from sklearn.decomposition import PCA

def normalize(vector):
    norm = np.sqrt(np.sum(vector ** 2))
    return vector / norm
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
    #print X.shape
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
def choose_img(src_dir,out_dir,num=1):
    filename_list = os.listdir(src_dir)  # read the directory files's name

    for i in range(70001,86385):
        imgName = os.path.join(src_dir, os.path.basename(filename_list[i]))
        if (os.path.splitext(imgName)[1] != '.jpg'): continue
        img = Image.open(imgName)

        img.save(out_dir + filename_list[i])
        print ("step is %d k",(i-70000)/1000)
    return 0


# rename_train_img_out = '/home/jains/datasets/gsndatasets/celebA/65536/'
# rename_test_img_out = '/home/jains/datasets/gsndatasets/celebA/2048_after_65536/'
def rename_file(data_dir,norm_data_dir):

    data_file_list = os.listdir(data_dir)#img数据名列表
    norm_data_list = os.listdir(norm_data_dir)#norm数据名列表
    data_file_list.sort()#两者文件名同时排序
    norm_data_list.sort()
    assert len(data_file_list) == len(norm_data_list),"图片文件数量和norm数据文件数量不全一致"#判断两个文件夹下文件个数是否一致

    for i in range(len(data_file_list)):
        assert data_file_list[i].split('.')[0] == norm_data_list[i].split('.')[0],"norm数据和img数据文件名不一致"
        imgName = os.path.join(data_dir, os.path.basename(data_file_list[i]))#img文件完全地址
        normName = os.path.join(norm_data_dir,os.path.basename(norm_data_list[i]))#norm文件完全地址

        # str1 = filename_list[j + (i / 16384) * 16384].split('.')

        if (os.path.splitext(imgName)[1] != '.jpg'): continue#检查文件后缀名是否为.jpg
        if (os.path.splitext(normName)[1] != '.npy'): continue#检查norm文件后缀名是否为.npy

        img = Image.open(imgName)#读取img图片
        norm = np.load(normName)#读取norm文件

        img.save('/home/jains/datasets/gsndatasets/celebA/2048_after_65536/' + str(i) + '.jpg')
        np.save('/home/jains/datasets/gsndatasets/celebA/2048_after_65536_ScatJ4_projected512_1norm/' + str(i) + '.npy',norm)
        print ("step is %d k"% (i / 1000))
# '/home/jains/datasets/gsndatasets/celebA/2048_after_65536/'
# rename_file(train_data_dir,rename_train_img_out)
# rename_file(test_data_dir,test_norm_dir)


def pca_data1(data_dir,out_dir):
    filename_list = os.listdir(data_dir)  # read the directory files's name
    filename_list.sort()#对读取的文件名进行排序
    # name = data_dir + filename_list[5068]
    # data = scio.loadmat(name)
    # tmp = scio.loadmat(data_dir + filename_list[5068])['all_buf']#读取.mat文件
    # np.save('./test_pcaname_list.npy',np.array(filename_list))#保存pca数据名字列表
    # number = len(filename_list)
    alldata1 = []
    for i in range(len(filename_list)):
        tmp = np.load(data_dir + '/' + filename_list[i])#读取npy文件
        # tmp = scio.loadmat(data_dir + '/' + filename_list[i])['all_buf']#读取.mat文件
        alldata1.append(tmp)

        if (i+1)%8192 == 0:
            print ('%d\n'%i)
            alldata = np.array(alldata1)
            alldata1 = []#释放内存
            alldata = alldata.reshape(len(alldata), -1)  # 16384*80064
            pca = PCA(n_components=512, copy=True, whiten=False, svd_solver='randomized')
            X = pca.fit_transform(alldata)  # 1024*512
            print (X.min(),X.max())
            # z = normalize(X[1])
            # x_or = pca.inverse_transform(X)#逆转换为原始数据
            for j in range(len(alldata)):
                str1 = filename_list[j + int(i/8192)*8192].split('.')
                np.save(out_dir +'/' + str1[0] + '.npy', normalize(X[j]))
            print (X.min(), X.max())
            # alldata = []

def pca_data(data_dir,out_dir):

    filename_list = os.listdir(data_dir)  # read the directory files's name
    filename_list.sort()#对读取的文件名进行排序
    # name = data_dir + filename_list[5068]
    # data = scio.loadmat(name)
    # tmp = scio.loadmat(data_dir + filename_list[5068])['all_buf']#读取.mat文件
    # np.save('./test_pcaname_list.npy',np.array(filename_list))#保存pca数据名字列表
    # number = len(filename_list)
    alldata1 = []
    for i in range(len(filename_list)):
        tmp = np.load(data_dir + '/' + filename_list[i])#读取npy文件
        # name = data_dir + filename_list[i]
        # t = scio.loadmat(name)
        # tmp = scio.loadmat(data_dir + filename_list[i])['all_buf']#读取.mat文件
        alldata1.append(tmp)
        fixed_value = 8192
        if (i+1)%fixed_value == 0:
            print ('%d\n'%i)
            alldata = np.array(alldata1)
            alldata1 = []#释放内存
            print(alldata.max(),alldata.min())
            alldata = alldata.reshape(len(alldata), -1)  # 16384*80064
            pca = PCA(n_components=512, copy=True, whiten=True)#, svd_solver='randomized')
            X = pca.fit_transform(alldata)  # 1024*512
            # origin_X = pca.inverse_transform(X)
            # print(np.count_nonzero(origin_X) / 8192,np.count_nonzero(X)/8192)
            print(X.max(),X.min())
            norms = np.sqrt(np.sum(X ** 2, axis=1))
            norms = np.expand_dims(norms, axis=1)
            norms = np.repeat(norms, 512, axis=1)
            X /= norms
            print(X.max(),X.min())
            for j in range(len(alldata)):
                str1 = filename_list[j + int(i/fixed_value)*fixed_value].split('.')
                np.save(out_dir +'/' + str1[0] + '.npy', X[j])


            # print ((X/(abs(X).max())).min(), (X/(abs(X).max())).max())
            # alldata = []



def scat_data(data_dir,outdata_dir,M,N,J):
    from kymatio import Scattering2D
    filename_list = os.listdir(data_dir)#read the directory files's name
    filename_list.sort()
    #np.save('./test_scatname_list.npy',np.array(filename_list))
    count = len(filename_list)
    scat = Scattering2D(J=J,shape=(M,N)).cuda()  # scattering transform
    for i in range(0,count):
        imgDir = os.path.join(data_dir, os.path.basename(filename_list[i]))
        if (os.path.splitext(imgDir)[1] != '.jpg'): continue

        # img = np.float16((np.array(Image.open(imgDir))/127.5) - 1.0)#与x保持一致，归一化到[-1,1]之间
        img = np.float16(np.array(Image.open(imgDir)))#与x保持一致，归一化到[-1,1]之间

        img = img.reshape(1,M,N,3)#1*128*128*3
        img_data = img.transpose(0, 3, 1, 2).astype(np.float32)  # 1*3*128*128
        img_data = torch.from_numpy(img_data).cuda()
        out_data = scat(img_data).cpu()#1*3*417*8*8

        str1 = filename_list[i].split('.')
        # str2 = imgName.split('.')

        np.save(outdata_dir +'/' + str1[0] + '.npy',out_data)
        if i%100 ==0:
            print ("step is %d"%i)
    return 0


def scat_data1(data_dir,outdata_dir,M,N,J):
    from kymatio import Scattering2D
    filename_list = os.listdir(data_dir)#read the directory files's name
    filename_list.sort()
    count = len(filename_list)

    scat = Scattering2D(J=J,shape=(M,N)).cuda()  # scattering transform
    batch_size = 256
    batch_image = []
    for count_idx in range(0,count):

        imgDir = os.path.join(data_dir, os.path.basename(filename_list[count_idx]))
        img = np.float32((np.array(Image.open(imgDir)) / 127.5)).transpose(2, 0, 1)
        batch_image.append(img)
        if((count_idx+1)%batch_size == 0):
            batch_image = torch.from_numpy(np.array(batch_image)).cuda()
            batch_scat = scat.forward(batch_image)
            batch_scat = batch_scat.cpu()

            for c in range(batch_size):
                img_scat = batch_scat[c]
                str1 = filename_list[c+(int(count_idx/batch_size))*batch_size].split('.')

                np.save(outdata_dir + '/' + str1[0] + '.npy', img_scat)

            batch_image = []
            print(count_idx)




    print("over")
    return 0



def zca_data(data_dir,out_dir):
    import keras
    filename_list = os.listdir(data_dir)  # read the directory files's name
    filename_list.sort()#对读取的文件名进行排序
    alldata1 = []
    for i in range(len(filename_list)):
        tmp = np.load(data_dir + '/' + filename_list[i])#读取npy文件
        tmp = tmp.reshape(tmp.shape[0]*tmp.shape[1],tmp.shape[2],tmp.shape[3])

        # name = data_dir + filename_list[i]
        # t = scio.loadmat(name)
        # tmp = scio.loadmat(data_dir + filename_list[i])['all_buf']#读取.mat文件

        alldata1.append(tmp)
        fixed_value = 8192
        if (i+1)%fixed_value == 0:
            print ('%d\n'%i)
            alldata = np.array(alldata1)
            alldata1 = []#释放内存
            print(alldata.max(),alldata.min())
            print(alldata.shape)

            zca = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,zca_whitening=True,
                                                         rescale=512/alldata.shape[1]/alldata.shape[2]/alldata.shape[3],data_format="channels_first")
            X=zca.fit(alldata)

            print(X.max(),X.min())
            norms = np.sqrt(np.sum(X ** 2, axis=1))
            norms = np.expand_dims(norms, axis=1)
            norms = np.repeat(norms, 512, axis=1)
            X /= norms
            print(X.max(),X.min())
            for j in range(len(alldata)):
                str1 = filename_list[j + int(i/fixed_value)*fixed_value].split('.')
                np.save(out_dir +'/' + str1[0] + '.npy', X[j])






train_data_dir = '/home/jains/datasets/gsndatasets/celebA/65536'
train_scat_dir = '/home/jains/datasets/gsndatasets/celebA/65536_ScatJ4'
train_norm_dir = '/home/jains/datasets/gsndatasets/celebA/65536_ScatJ4_projected512_1norm'

test_data_dir = '/home/jains/datasets/gsndatasets/celebA/2048_after_65536'
test_scat_dir = '/home/jains/datasets/gsndatasets/celebA/2048_after_65536_ScatJ4'
test_norm_dir = '/home/jains/datasets/gsndatasets/celebA/2048_after_65536_ScatJ4_projected512_1norm'
# outimg_dir = '/home/jains/datasets/gsn/celebA_128/65536/'
M = 128;N = 128;J = 4
#函数调用
# scat_data1(test_data_dir,test_scat_dir,M,N,J)
# scat_data1(train_data_dir,train_scat_dir,M,N,J)
pca_data1(test_scat_dir,test_norm_dir)
pca_data1(train_scat_dir,train_norm_dir)


# pca_data('/home/jains/datasets/gsndatasets/debugdata/65536_ScatJ4','/home/jains/datasets/gsndatasets/debugdata/65536_ScatJ4_projected512_1norm')

# make_norm_data(outputdata_dir)
# cut_celeba_face()
# face_dir = '/home/jains/datasets/celebA/cut_face/'
# choose_img(face_dir,test_data_dir,1)


