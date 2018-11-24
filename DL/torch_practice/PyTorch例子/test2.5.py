# -*- coding: utf-8 -*-

#--------------------------------------pytorch:nn---------------------一种封装的模块，类似于keras

#权重不需要自己定义，内部存在权重，可以直接使用

import torch
from torch.autograd import Variable

# N 批量大小; D_in是输入尺寸;
# H是隐藏尺寸; D_out是输出尺寸.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量来保存输入和输出,并将它们包装在变量中.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# 使用nn包将我们的模型定义为一系列图层.
# nn.Sequential是一个包含其他模块的模块,并将它们按顺序应用以产生其输出.
# 每个线性模块使用线性函数计算来自输入的输出,并保存内部变量的权重和偏差.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)


# nn包还包含流行损失函数的定义;
# 在这种情况下,我们将使用均方差(MSE)作为我们的损失函数.
loss_fn = torch.nn.MSELoss(size_average=False)


learning_rate = 1e-4
for t in range(500):
    # 正向传递:通过将x传递给模型来计算预测的y.
    # 模块对象会覆盖__call__运算符,因此您可以将它们称为函数.
    # 这样做时,您将输入数据的变量传递给模块,并生成输出数据的变量.
    y_pred = model(x)

    # 计算和打印损失.
    # 我们传递包含y的预测值和真值的变量,并且损失函数返回包含损失的变量.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # 在运行反向传递之前将梯度归零.
    model.zero_grad()

    # 向后传递:计算相对于模型的所有可学习参数的损失梯度.
    # 在内部,每个模块的参数都存储在变量require_grad = True中,
    # 因此该调用将计算模型中所有可学习参数的梯度.
    loss.backward()

    # 使用梯度下降更新权重.
    # 每个参数都是一个变量,所以我们可以像我们以前那样访问它的数据和梯度.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data