#coding=utf-8

import os
from utils import create_folder, normalize
from generator_architecture import Generator, weights_init
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np

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
        self.nb_epochs_to_save = 1#几个epoch保存一次

    def train(self,epoch_to_restore = 0):
        g = Generator(self.nb_channels_first_layer,self.dim)#生成器网络

        # if epoch_to_restore > 0:
        #     filename_model = os.path.join(self.dir_models,'epoch_{}.pth'.format(epoch_to_restore))
        #     g.load_state_dict(torch.load(filename_model))
        # else:
        #     g.apply(weights_init)

        # g.cuda()
        g.train()

        dataset = EmbeddingsImagesDataset(self.dir_z_train,self.dir_x_train)
        dataloader = DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True,num_workers=4,pin_memory=True)

        fixed_dataloader = DataLoader(dataset,16)#选取16个作为验证集，测试验证数据

        fixed_batch = next(iter(fixed_dataloader))#所有值/127.5 - 1

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
                g.eval()#验证集?????????????????
                g_z = g.forward(z)
                images = make_grid(g_z.data[:16],nrow=4,normalize=True)#把若干幅图拼接为一副图

                images_tmp = images.cpu().numpy().transpose((1,2,0))
                Image.fromarray(np.unit8((images_tmp + 1)*127.5)).save('Users/jains/datasets/test/' + str(epoch) + '.jpg')#存储验证集中重构出的图片



                writer.add_image('generations', images, epoch)
                filename = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
                torch.save(g.state_dict(), filename)
        finally:

            print ('[*] Closing Writer')
            writer.close()
    def generate_from_model(self,epoch):
        filename_model = os.path.join(self.dir_models,'epoch_{}.pth'.format(epoch))
        g = Generator(self.nb_channels_first_layer,self.dim)#生成器网络
        g.load_state_dict(torch.load(filename_model))#加载存储的模型????????????
        # g.cuda()
        g.eval()

        '''从已有模型中重构出训练图片和测试图片'''
        def _generate_from_model(dir_z,dir_x,train_test):
            dataset = EmbeddingsImagesDataset(dir_z,dir_x)#?????????????
            fixed_dataloader = DataLoader(dataset=dataset,batch_size=16)#验证集
            fixed_batch = next(iter(fixed_dataloader))

            z = Variable(fixed_batch['z']).type(torch.FloatTensor)
            g_z = g.forward(z)
            filename_images = os.path.join(self.dir_experiment,'epoch_{}_{}.png'.format(epoch,train_test))#生成的图片的名称
            temp = make_grid(g_z.data[:16],nrow=4).cpu().numpy().transpose((1,2,0))
            Image.fromarray(np.unint8((temp + 1)*127.5)).save(filename_images)#保存图像到本地

        _generate_from_model(self.dir_z_train,self.dir_x_train,'train')#从训练集中生成图片
        _generate_from_model(self.dir_z_test,self.dir_x_test,'test')#从测试集中生成图片


        ''''''
        def _generate_path(dir_z,dir_x,train_test):
            dataset = EmbeddingsImagesDataset(dir_z,dir_x)#?????????????
            fixed_dataloader = DataLoader(dataset,2,shuffle=True)
            fixed_batch = next(iter(fixed_dataloader))

            z0 = fixed_batch['z'][[0]].numpy()
            z1 = fixed_batch['z'][[1]].numpy()

            batch_z = np.copy(z0)
            nb_samples = 100
            interval = np.linspace(0,1,nb_samples)
            for t in interval:
                if t>0:
                    zt = normalize((1 - t) * z0 + t * z1)
                    batch_z = np.vstack((batch_z,zt))

            z = Variable(torch.fromnumpy(batch_z)).type(torch.FloatTensor)#非cuda版本
            g_z = g.forward(z)#前向传播
            g_z = g_z.data.cpu().numpy().transpose((0,2,3,1))#转换为非torch类型数据
            folder_to_save = os.path.join(self.dir_experiment,'epoch_{}_{}_path'.format(epoch,train_test))#保存图片的文件夹
            create_folder(folder_to_save)#创建文件夹

            for idx in range(nb_samples):
                filename_image = os.path.join(folder_to_save,'{}.png'.format(idx))
                Image.fromarray(np.unint8((g_z[idx] + 1) * 127.5)).save(filename_image)#保存图片

        _generate_path(self.dir_z_train,self.dir_x_train,'train')
        _generate_path(self.dir_z_test,self.dir_x_test,'test')

        def _generate_random():
            nb_samples = 16
            z = np.random.randn(nb_samples,self.dim)#随机抽样样本
            norms = np.sqrt(np.sum(z ** 2,axis=1))
            norms = np.expand_dims(norms,axis=1)
            norms = np.repeat(norms,self.dim,axis=1)
            z /= norms#随机获取的z

            z = Variable(torch.fromnumpy(z)).type(torch.FloatTensor)#GPU转为cuda()
            g_z = g.forward(z)
            filename_images = os.path.join(self.dir_experiment,'epoch_{}_random.png'.format(epoch))#定义随机z生成图片名称
            temp = make_grid(g_z.data[:16],nrow=4).cpu().numpy().transpose((1,2,0))#转为可以存储的图片数据
            Image.fromarray(np.unint8((temp + 1) * 127.5)).save(filename_images)#保存图片

        _generate_random()#调用该函数

















from utils import create_name_experiment
# from GSN import GSN

parameters = dict()
parameters['dataset'] = 'celebA_128'
parameters['train_attribute'] = '65536'
parameters['test_attribute'] = '2048_after_65536'
parameters['dim'] = 512
parameters['embedding_attribute'] = 'ScatJ4_projected{}_1norm'.format(parameters['dim'])
parameters['nb_channels_first_layer'] = 32#网络第一层节点个数

parameters['name_experiment'] = create_name_experiment(parameters, 'NormL1')
gsn = GSN(parameters)
gsn.train()