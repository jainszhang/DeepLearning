#-*- coding:utf-8 -*-


#-----------------------------------------神经网络--------------------------
#autograd 实现了反向传播功能, 但是直接用来写深度学习的代码在很多情况下还是稍显复杂, torch.nn 是专门为神经网络设计的模块化接口. nn 构建于 Autograd 之上, 可用来定义和运行神经网络.
# nn.Module 是 nn 中最重要的类, 可把它看成是一个网络的封装, 包含网络各层定义以及 forward 方法, 调用 forward(input) 方法, 可返回前向传播的结果.
#主要有torch.nn,autograd,torch.nn,nn.Module,forward

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):#从nn中继承而来
    def __init__(self):
        super(Net,self).__init__()#super() 函数是用于调用父类(超类)的一个方法。用于多重继承

        self.conv1 = nn.Conv2d(1,6,5)## 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数, '5'表示卷积核为5*5
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16 * 5 * 5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)# 如果大小是正方形, 则只能指定一个数字
        x = x.view(-1,self.num_flat_features(x))#卷积层需要展开为全链接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        # print(list(x.size()))
        size = np.array(list(x.size()))# 除批量维度外的所有维度
        print(size)
        num_features = 1
        for s in size:
            num_features *= s
        print("numfeatures",num_features)
        return num_features

net = Net()
# print(net)
#你只要在 nn.Module 的子类中定义了 forward 函数, backward 函数就会自动被实现(利用 autograd ). 在 forward 函数中可使用任何 Tensor 支持的操作
# 网络的可学习参数通过 net.parameters() 返回, net.named_parameters 可同时返回学习的参数以及名称.

params = list(net.parameters())
# print(list(net.named_parameters()))
# print(len(params))
print(params[0].size())

input = Variable(torch.randn(1,1,32,32))
out = net(input)
# print(out)

#网络中所有参数清零
net.zero_grad()
out.backward(torch.randn(1,10))

# torch.nn 只支持小批量(mini-batches), 不支持一次输入一个样本, 即一次必须是一个 batch.
# 例如, nn.Conv2d 的输入必须是 4 维的, 形如 nSamples x nChannels x Height x Width.
# 如果你只想输入一个样本, 需要使用 input.unsqueeze(0) 将 batch_size 设置为 1.





#-----------------------------------------------------损失函数-------------------------------------------
#nn.MSELoss表示均方误差
output = net(input)
target = Variable(torch.arange(1,11))
criterion = nn.MSELoss()
loss = criterion(output,target)
print(loss)
# print(loss.grad_fn)





#-----------------------------------------------------反向传播------------------------------------------
#为了反向传播误差, 我们所要做的就是 loss.backward(). 你需要清除现有的梯度, 否则梯度会累加之前的梯度.
net.zero_grad()
# print(net.conv1.bias.grad)
loss.backward()
# print(net.conv1.bias.grad)









#-----------------------------------------------更新权重--------------------------------------------
#weight = weight - learning_rate * gradient
learning = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning)
import torch.optim as optim
opt = optim.SGD(net.parameters(),lr = 0.01)# 新建一个优化器, 指定要调整的参数和学习率

opt.zero_grad()# 首先梯度清零
output = net(input)
loss = criterion(output,target)
loss.backward()
opt.step()# 更新参数
print (loss)