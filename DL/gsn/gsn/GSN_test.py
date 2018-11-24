#coding=utf-8

import os
from utils import create_folder, normalize
from generator_architecture import Generator, weights_init
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torchvision.utils import make_grid
import torch.optim as optim


from EmbeddingsImagesDataset import EmbeddingsImagesDataset

class GSN:
    def __init__(self,parameters):
        dir_datasets = os.path.expanduser('/Users/jains/datasets/gsndatasets')#把path中包含的"~"和"~user"转换成用户目录
        dir_experiments = os.path.expanduser('./experiments')

        dataset = parameters['dataset']#the directory of dataset
        train_attribute = parameters['train_attribute']
        test_attribute = parameters['test_attribute']
        embedding_attribute = parameters['embedding_attribute']

        print('dir_datasets:{},dataset:{},train_attribute:{},test_attribute:{},embedding_attribute:{}'.format(dir_datasets,dataset,train_attribute,test_attribute,embedding_attribute))


        self.dim = parameters['dim']
        self.nb_channels_first_layer = parameters['nb_channels_first_layer']

        name_experiment = parameters['name_experiment']#experiments' name

        self.dir_x_train = os.path.join(dir_datasets,dataset,'{0}'.format(train_attribute))#the last directory of train data
        self.dir_x_test = os.path.join(dir_datasets,dataset,'{0}'.format(test_attribute))
        self.dir_z_train = os.path.join(dir_datasets,dataset,'{0}_{1}'.format(train_attribute,embedding_attribute))
        self.dir_z_test = os.path.join(dir_datasets, dataset, '{0}_{1}'.format(test_attribute, embedding_attribute))


        self.dir_experiment = os.path.join(dir_experiments,'gsn_hf',name_experiment)
        self.dir_models = os.path.join(self.dir_experiment, 'models')
        self.dir_logs = os.path.join(self.dir_experiment, 'logs')
        create_folder(self.dir_models)#create experiments' folder
        create_folder(self.dir_logs)

        self.batch_size = 32
        self.nb_epochs_to_save = 1#????????

    def train(self,epoch_to_restore = 0):
        g = Generator(self.nb_channels_first_layer,self.dim)#generator network

        # if epoch_to_restore > 0:
        #     filename_model = os.path.join(self.dir_models,'epoch_{}.pth'.format(epoch_to_restore))
        #     g.load_state_dict(torch.load(filename_model))
        # else:
        #     g.apply(weights_init)

        # g.cuda()
        g.train()

        dataset = EmbeddingsImagesDataset(self.dir_z_train,self.dir_x_train)
        dataloader = DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True,num_workers=4,pin_memory=True)

        fixed_dataloader = DataLoader(dataset,16)#??????????????????
        fixed_batch = next(iter(fixed_dataloader))

        criterion = torch.nn.L1Loss()#定义损失函数
        optimizer = optim.Adam(g.parameters())
        writer = SummaryWriter(self.dir_logs)#写入日志文档，参数为文件夹名字
        try:
            epoch = epoch_to_restore
            while True:
                g.train()
                for _ in range(self.nb_epochs_to_save):
                    epoch = epoch + 1
                    print("epoch is %d"%epoch)
                    for idx_batch,current_batch in enumerate(tqdm(dataloader)):
                        g.zero_grad()#设置梯度为0
                        x = Variable(current_batch['x']).type(torch.FloatTensor)
                        z = Variable(current_batch['z']).type(torch.FloatTensor)
                        g_z = g.forward(z)#前向传播

                        loss = criterion(g_z,x)#计算损失
                        loss.backward()#反向传播
                        optimizer.step()#更新参数
                    writer.add_scalar('train_loss',loss,epoch)#写入图像名称，loss数字，迭代次数
                z = Variable(fixed_batch['z']).type(torch.FloatTensor)
                g.eval()#验证集
                g_z = g.forward(z)
                images = make_grid(g_z.data[:16],nrow=4,normalize=True)#把若干幅图拼接为一副图


                writer.add_image('generations', images, epoch)
                filename = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
                torch.save(g.state_dict(), filename)
        finally:

            print ('[*] Closing Writer')
            writer.close()





from utils import create_name_experiment
# from GSN import GSN

parameters = dict()
parameters['dataset'] = 'celebA_128'
parameters['train_attribute'] = '65536'
parameters['test_attribute'] = '2048_after_65536'
parameters['dim'] = 512
parameters['embedding_attribute'] = 'ScatJ4_projected{}_1norm'.format(parameters['dim'])
parameters['nb_channels_first_layer'] = 32##????????????????

parameters['name_experiment'] = create_name_experiment(parameters, 'NormL1')
# print (parameters['name_experiment'])
gsn = GSN(parameters)
gsn.train(1)