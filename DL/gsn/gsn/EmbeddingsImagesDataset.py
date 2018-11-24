#coding=utf-8
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import get_nb_files


class EmbeddingsImagesDataset(Dataset):
    def __init__(self, dir_z, dir_x, nb_channels=3):
        # assert出现错误条件时就崩溃（返回错误）
        assert get_nb_files(dir_z) == get_nb_files(dir_x)#如果x个数和z对应个数不一致，则程序错误
        assert nb_channels in [1, 3]#如果通道数不在1-3之内，则程序错误

        self.nb_files = get_nb_files(dir_z)#获取z的个数

        self.nb_channels = nb_channels#获取通道数

        self.dir_z = dir_z
        self.dir_x = dir_x

    def __len__(self):#返回文件个数
        return self.nb_files

    def __getitem__(self, idx):
        filename = os.path.join(self.dir_z, '{}.npy'.format(idx))#获取z的文件名
        z = np.load(filename)#加载.npy文件

        filename = os.path.join(self.dir_x, '{}.jpg'.format(idx))#获取图片x的文件名
        if self.nb_channels == 3:
            x = (np.ascontiguousarray(Image.open(filename), dtype=np.uint8).transpose((2, 0, 1)) / 127.5) - 1.0#三通道情况处理，全部除127.5 - 1
        else:
            x = np.expand_dims(np.ascontiguousarray(Image.open(filename), dtype=np.uint8), axis=-1)#单通道或者2通道情况处理
            x = (x.transpose((2, 0, 1)) / 127.5) - 1.0

        sample = {'z': z, 'x': x}
        return sample