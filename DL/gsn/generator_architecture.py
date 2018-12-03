# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

import os

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from EmbeddingsImagesDataset import EmbeddingsImagesDataset

import numpy as np
import torch

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)


class Generator(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=4):
        super(Generator, self).__init__()

        nb_channels_input = nb_channels_first_layer * 32

        self.main = nn.Sequential(
            nn.Linear(in_features=z_dim,        #z_dim = 512
                      out_features=size_first_layer * size_first_layer * nb_channels_input,
                      bias=False),
            View(-1, nb_channels_input, size_first_layer, size_first_layer),
            nn.BatchNorm2d(nb_channels_input, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),

            ConvBlock(nb_channels_input, nb_channels_first_layer * 16, upsampling=True),
            ConvBlock(nb_channels_first_layer * 16, nb_channels_first_layer * 8, upsampling=True),
            ConvBlock(nb_channels_first_layer * 8, nb_channels_first_layer * 4, upsampling=True),
            ConvBlock(nb_channels_first_layer * 4, nb_channels_first_layer * 2, upsampling=True),
            ConvBlock(nb_channels_first_layer * 2, nb_channels_first_layer, upsampling=True),

            ConvBlock(nb_channels_first_layer, nb_channels_output=3, tanh=True)
        )

    def forward(self, input_tensor):
        # print ("original tensor size is ",input_tensor.size())
        # print ("out tensor size is ",self.main(input_tensor))
        return self.main(input_tensor)


class ConvBlock(nn.Module):
    def __init__(self, nb_channels_input, nb_channels_output, upsampling=False, tanh=False):
        super(ConvBlock, self).__init__()

        self.tanh = tanh
        self.upsampling = upsampling

        filter_size = 7
        padding = (filter_size - 1) // 2

        if self.upsampling:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(nb_channels_input, nb_channels_output, filter_size, bias=False)
        self.bn_layer = nn.BatchNorm2d(nb_channels_output, eps=0.001, momentum=0.9)

    def forward(self, input_tensor):
        if self.upsampling:
            output = self.up(input_tensor)
        else:
            output = input_tensor

        output = self.pad(output)
        output = self.conv(output)
        output = self.bn_layer(output)

        if self.tanh:
            output = F.tanh(output)
        else:
            output = F.relu(output)

        return output


def weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)





import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),      #(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  #output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

dummy_input = Variable(torch.rand(13, 1, 28, 28)) #假设输入13张1*28*28的图片
model = LeNet()
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input, ))



















# if __name__ == '__main__':
#     dir_datasets = os.path.expanduser('/home/jains/datasets/gsndatasets')
#     dataset = 'celebA_128_1k'
#     dataset_attribute = '65536'
#     embedding_attribute = 'ScatJ4_projected512_1norm'
#
#
#     dir_x_train = os.path.join(dir_datasets, dataset, '{0}'.format(dataset_attribute))#原始图片文件夹
#     dir_z_train = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(dataset_attribute, embedding_attribute))#scatj4数据文件夹
#
#     dataset = EmbeddingsImagesDataset(dir_z_train, dir_x_train)
#     fixed_dataloader = DataLoader(dataset, batch_size=2)
#     fixed_batch = next(iter(fixed_dataloader))
#
#     nb_channels_first_layer = 32#第一层channel个数
#
#
#     input_tensor = Variable(fixed_batch['z']).cuda()#tensor转换为view类型数据
#     g = Generator(nb_channels_first_layer, 512)#
#
#     from tensorboardX import SummaryWriter
#     with SummaryWriter(comment='Generator') as w:
#         w.add_graph(g,(input_tensor))
#
#     # g.cuda()
#     # g.train()
#
#     # output = g.forward(input_tensor)
#     # save_image(output[:16].data, 'temp.png', nrow=4)#保存图片
