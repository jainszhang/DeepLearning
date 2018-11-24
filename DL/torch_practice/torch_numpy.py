# -*- coding: utf-8 -*-


import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)




import torch
import torch.nn.functional as F
#


#----------------------------------------------两种搭建网络的方式--------------------------------------

class Net(torch.nn.Module):#继承torch的model
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()#继承init的功能
        self.hidden = torch.nn.Linear(n_feature,n_hidden)#隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden,n_output)#输出层线性输出

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1,n_hidden=10,n_output=1)#定义网络的对象


#第二种网络方式，不用像第一种那样自己定义，但是如果需要自定义结构，例如RNN就需要上述的方式了
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# print(net)
#训练网络
opt = torch.optim.SGD(net.parameters(),lr=0.2)#优化器定义
loss_func = torch.nn.MSELoss()#损失函数

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction,y)#计算两者误差
    opt.zero_grad()#清空上一步残余更新误差
    loss.backward()#误差的反响传播，计算参数更新值
    opt.step()#将参数更新值加到net的parametters上
    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)



#--------------------------------------------------------------------保存和提取网络------------------------------
def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # 训练
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #两种方式保存网络
    torch.save(net1,'net.pkl')#保存整个网络
    torch.save(net.state_dict(),'net_params.pkl')#只保存网络中的参数（速度快，内存小）




    ########提取网络##
#提取整个网络
def restore_net(x):
    net2 = torch.load('net.pkl')
    prediction = net2(x)

#只提取网络参数
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    #将保存的参数复制导net3当中
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)



#------------------------------------------------------------------------------------批训练中的DataLoader------------------------------------
import torch.utils.data as Data
BATCH_SIZE = 5      # 批训练的数据个数
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)# 先转换成 torch 能识别的 Dataset
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,#多线程读取数据
)
for epoch in range(3):   # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...

        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())



#-------------------------------------------------------------------------------------------------使用GPU加速训练----------------------------------
import torchvision

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 第一步：：：：!!!!!!!! 修改 test data 形式 !!!!!!!!! #
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()





#第二步：：：：把CNN参数变成GPU兼容模式
class CNN(torch.nn.Module):
    # ...

    cnn = CNN()
# !!!!!!!! 转换 cnn 去 CUDA !!!!!!!!! #
    cnn.cuda()      # Moves all model parameters and buffers to the GPU.






#第三步：：：：在 train 的时候, 将每次的training data 变成 GPU 形式. + .cuda()
for epoch ..:
    for step, ...:
        # !!!!!!!! 这里有修改 !!!!!!!!! #
        b_x = x.cuda()    # Tensor on GPU
        b_y = y.cuda()    # Tensor on GPU

        ...

        if step % 50 == 0:
            test_output = cnn(test_x)

            # !!!!!!!! 这里有修改  !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # 将操作放去 GPU

            accuracy = torch.sum(pred_y == test_y) / test_y.size(0)
            ...

test_output = cnn(test_x[:10])

# !!!!!!!! 这里有修改 !!!!!!!!! #
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # 将操作放去 GPU
...
print(test_y[:10], 'real number')