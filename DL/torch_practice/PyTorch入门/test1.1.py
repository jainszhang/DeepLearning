# -*- coding: utf-8 -*-

from __future__ import print_function
import torch


#-----------------------------------tensor加法操作-------------------
x = torch.Tensor(5,3)
print(x)
x = torch.rand(5,3)
print(x)
# print(x.size())
y = torch.rand(5,3)
print(x + y)

print(torch.add(x,y))

result = torch.Tensor(5,3)
torch.add(x,y,out=result)
print(result)
y.add_(x)
print(y)
#任何改变张量的操作方法都是以后缀 _ 结尾的. 例如: x.copy_(y), x.t_(), 将改变张量 x.
print(y[:,1])

#改变tensor大小
x = torch.randn(4,4)#randn存在正负，rand仅仅式正随机数
y  = x.view(16)
z = x.view(-1,8)
print(x)
print(y)
print(z)


#-------------------------------------------------------------------------------转换Numpy数组为Torch Tensor--------------------------------------
#除了 CharTensor 之外, CPU 上的所有 Tensor 都支持与Numpy进行互相转换
print("----------------------------------------")
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out = a)
print(a)
print(b)





#--------------------------------------------------------------------------------CUDA Tensor--------------------------------------------
#可以使用 .cuda 方法将 Tensors 在GPU上运行.
if torch.cuda.is_available():#若cuda可用
    x = x.cuda()
    y = y.cuda()
    print(torch.add(x,y))