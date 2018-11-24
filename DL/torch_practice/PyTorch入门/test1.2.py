# -*- coding: utf-8 -*-



from __future__ import print_function
import torch

#-----------------------------------------------------------------自动求导求微分------------------------------
#Variable 和 Function 是相互联系的, 并且它们构建了一个非循环的图, 编码了一个完整的计算历史信息. 每一个 variable（变量）都有一个 .grad_fn 属性,
# 它引用了一个已经创建了 Variable 的 Function （除了用户创建的 Variable `` 之外 - 它们的 ``grad_fn is None ）.
#使用autograd，主要有以下几个函数：.Variable,.backward,.data,Function类
from torch.autograd import Variable

x = Variable(torch.ones(2,2),requires_grad = True)
print(x)
y = x + 2
print(y)
z  = y * y * 3
out = z.mean()
print(out)







#-----------------------------------------------------------------梯度-----------------------------------------
#我们现在开始了解反向传播, out.backward() 与 out.backward(torch.Tensor([1.0])) 这样的方式一样

out.backward()
print(x.grad)

x = torch.randn(3)
x = Variable(x,requires_grad = True)
y = x * 2
while y.data.norm <1000:
    y = y*2
print("y",y)


gradients = torch.FloatTensor([0.1,1.0,0.001])
y.backward(gradients)
print(x.grad)