# -*- coding:utf-8 -*-

#---------------------------------数据并行-----------------------------
#多GPU使用DataParallel
# model.gpu()#分配GPU

# mytensor  =my_tensor.gpu()#随后, 你可以将你的所有张量拷贝到上面的GPU:
# 此处请注意: 如果只是调用 mytensor.gpu() 是不会将张量拷贝到 GPU 的.你需要将它赋给一个 新的张量, 这个张量就能在 GPU 上使用了.

#可以通过多个GPU并行运行模型
# model = nn.DataParallel(model)


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# 1：参数和数据加载
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


#2：伪数据集
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



rand_loader = DataLoader(dataset=RandomDataset(input_size, 100),
                         batch_size=batch_size, shuffle=True)


#3：简单模型
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())

        return output



#4：创建模型和DataParallel
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

if torch.cuda.is_available():
   model.cuda()




#5：运行模型
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)

    output = model(input_var)
    print("Outside: input size", input_var.size(),
          "output_size", output.size())

# 6：总结
# DataParallel 自动地将数据分割并且将任务送入多个GPU上的多个模型中进行处理. 在每个模型完成任务后, DataParallel 采集和合并所有结果, 并将最后的结果呈现给你.

