#coding=utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.models.nnlib import concat, weight_variable_cpu, batch_norm
from resnet.models.model_factory import RegisterModel
from resnet.utils import logger

log = logger.get()




@RegisterModel("resnet")
class ResNetModel(object):

    def __init__(self,
                 config,
                 is_training=True,#train or eval
                 inference_only=False,#build optimizer or not
                 inp=None,#input
                 label=None,
                 dtype=tf.float32,
                 batch_size=None,
                 apply_grad=True,
                 idx=0):
        """ResNet constructor.

        Args:
          config: Hyperparameters.
          is_training: One of "train" and "eval".
          inference_only: Do not build optimizer.
        """
        self._config = config
        self._dtype = dtype
        self._apply_grad = apply_grad
        self._saved_hidden = []
        # Debug purpose only.
        self._saved_hidden2 = []
        self._bn_update_ops = []
        self.is_training = is_training
        self._batch_size = batch_size
        self._dilated = False


        #输入----目的是为了把申请变量和训练时放在一起
        if inp is None:
            x = tf.placeholder(
                dtype=dtype,shape=[batch_size,config.height,config.width,config.num_channel],
                name="x")
        else:
            x = inp

        if label in None:
            y = tf.placeholder(dtype=tf.int32,shape=[batch_size],name="y")
        else:
            y = label
        logits = self.build_inference_network(x)



    def _conv(self,name,x,filter_size,in_filters,out_filters,strides):
        """做卷积"""
        with tf.variable_scope(name):
            if self.config.filter_initialization == "normal":
                n = filter_size * filter_size *out_filters
                init_method = "truncated_normal"
                init_param = {"mean":0,"stddev":np.sqrt(2./n)}
            elif self.config.filter_initialization == "uniform":
                init_method = "uniform_scaling"
                init_param = {"factor":1.}
            kernel = self._weight_variable(
                [filter_size, filter_size, in_filters, out_filters],
                init_method=init_method,
                init_param=init_param,
                wd=self.config.wd,
                dtype=self.dtype,
                name="w")

            return tf.nn.conv2d(x,kernel,strides,padding="SAME")

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype = tf.float32,
                         init_param = None,
                         wd = None,
                         name = None,
                         trainable=True,
                         seed=0):
        return weight_variable_cpu(
            shape,
            init_method=init_method,
            dtype = dtype,
            init_param=init_param,
            wd=wd,
            name=name,
            trainable=trainable,
            seed=seed
        )

    #定义归一化层
    def _batch_norm(self,name,x,add_ops=True):
        with tf.variable_scope(name):
            n_out = x.get_shape()[-1]
            try:
                n_out = int(n_out)
                shape = [n_out]
            except:
                shape = None
            beta = self._weight_variable(
                shape,
                init_method = "constant",
                init_param = {"val":0.0},
                name = "beta",
                dtype = self.dtype
            )
            gamma = self._weight_variable(
                shape,
                init_method = "constant",
                init_param = {"val":1.},
                name = "gamma",
                dtype = self.dtype
            )
            normed,ops = batch_norm(
                x,
                self.is_training,
                gamma=gamma,
                beta=beta,
                axes=[0,1,2],
                eps=1e-3,
                name="bn_out"
            )
            if add_ops:
                if ops is not None:
                    self._bn_update_ops.extend(ops)
            return normed

    #定义relu层
    def _relu(self,name,x):
        return tf.nn.relu(x,name=name)

    #定义卷积层
    def _init_conv(self,x,n_filters):
        '''创建卷积层'''
        config = self.config
        init_filter = config.init_filter
        with tf.variable_scope("init"):
            h = self._conv("init_conv",x,init_filter,self.config.num_channels,#conv层
                           n_filters,self._stride_arr(config.init_strides))
            h = self._batch_norm("init_bn",h)#bn层
            h = self._relu("init_relu",h)#relu层

            if config.init_max_pool:
                h = tf.nn.max_pool(h,[1,3,3,1],[1,2,2,1],"SAME")#polling层
        return h

    """输入步长，输出步长矩阵"""
    def _stride_arr(self,stride):
        return [1,stride,stride,1]

    def _bottleneck_residual_inner(self,
                                   x,
                                   in_filter,
                                   out_filter,
                                   stride,
                                   no_activation=False,
                                   add_bn_ops = True):
        with tf.variable_scope("sub1"):
            if not no_activation:
                x = self._batch_norm("bn1",x,add_ops=add_bn_ops)
                x = self._relu("relu1",x)
            x = self._conv("conv1",x,1,in_filter,out_filter // 4,stride)#//表示除法向下取整
        with tf.variable_scope("sub2"):
            x = self._batch_norm("bn2",x,add_ops=add_bn_ops)
            x = self._relu("relu2",x)
            x = self._conv("conv2",x,3,out_filter // 4,out_filter // 4,self._stride_arr(1))

        with tf.variable_scope("sub3"):
            x = self._batch_norm("bn3",x,add_ops=add_bn_ops)
            x = self._relu("relu3",x)
            x = self._conv("conv3",x,1,out_filter // 4,out_filter,self._stride_arr(1))#??????????
        return x

    def _possible_bottleneck_downsample(self,x,in_filter,out_filter,stride):
        if stride[1] > 1 or in_filter != out_filter:
            x = self._conv("projcet",x,1,in_filter,out_filter,stride)
        return x
    def _possible_downsample(self,x,in_filter,out_filter,stride):
        """Downsample the feature map using average pooling, if the filter size
            does not match."""
        if stride[1] > 1:
            with tf.variable_scope("downsample"):
                x = tf.nn.avg_pool(x,stride,stride,"VALID")
        if in_filter < out_filter:
            with tf.variable_scope("pad"):
                x = tf.pad(x,
                           [[0,0],[0,0],[0,0],[(out_filter-in_filter) // 2,(out_filter-in_filter) // 2]])
        return x
    #定义residual块
    def _bottleneck_residual(self,
                             x,
                             in_filter,
                             out_filter,
                             stride,
                             no_activation=False,
                             add_bn_ops = True):
        orig_x = x
        x = self._bottleneck_residual_inner(#??????????????????????
            x,
            in_filter,
            out_filter,
            stride,
            no_activation=no_activation,
            add_bn_ops=add_bn_ops
        )
        x += self._possible_bottleneck_downsample(orig_x,in_filter,out_filter,stride)#????????????
        return x

    def _residual_inner(self,
                        x,
                        in_filter,
                        out_filter,
                        stride,
                        no_activation=False,
                        add_bn_ops = True):
        with tf.variable_scope("sub1"):
            if not no_activation:
                x = self._batch_norm("bn1",x,add_ops=add_bn_ops)
                x = self._relu("relu1",x)
            x = self._conv("conv1",x,3,in_filter,out_filter,stride)

        with tf.variable_scope("sub2"):
            x = self._batch_norm("bn2",x,add_ops=add_bn_ops)
            x = self._relu("relu2",x)
            x = self._conv("conv2",x,3,out_filter,out_filter,[1,1,1,1])
        return x

    def _residual(self,
                  x,
                  in_filter,
                  out_filter,
                  stride,
                  no_activation=False,
                  add_bn_ops=True):
        """Residual unit with 2 sub layers."""
        orig_x = x
        x = self._residual_inner(#???????????????????
            x,
            in_filter=in_filter,
            out_filter=out_filter,
            stride=stride,
            no_activation=no_activation,
            add_bn_ops=add_bn_ops

        )
        x += self._possible_downsample(orig_x,in_filter,out_filter,stride)#?????????????
        return x

    def _global_avg_pool(self,x):

        return tf.reduce_mean(x,[1,2])

    def _fully_connected(self,x,out_dim):
        x_shape = x.get_shape()
        d = x_shape[1]
        w = self._weight_variable(
            [d,out_dim],
            init_method="uniform_scaling",
            init_param={"factor":1.0},
            wd = self.config.wd,
            dtype = self.dtype,
            name="w"
        )

        b = self._weight_variable(
            [out_dim],
            init_method="constant",
            init_param={"val":0.},
            name = "b",
            dtype = self.dtype
        )
        return tf.nn.xw_plus_b(x,w,b)

    def build_inference_network(self,x):
        config = self.config
        is_training = self.is_training
        num_stages = len(self.config.num_residual_units)
        strides = config.strides
        activate_before_residual = config.activate_before_residual
        filters = [ff for ff in config.filters]#复制config中的filters大小
        h = self._init_conv(x,filters[0])#卷积层,第一个卷积层所需核个数
        if config.use_bottleneck:
            res_func = self._bottleneck_residual#？？？？？？？？？？？？？？？？？？？？？？？？
            # For CIFAR-10 it's [16, 16, 32, 64] => [16, 64, 128, 256]-- 通道个数增加
            for ii in range(1,len(filters)):
                filters[ii] *=4
        else:
            res_func = self._residual

        # New version, single for-loop. Easier for checkpoint.
        #循环建立residual块
        nlayers = sum(config.num__residual_units)
        ss = 0
        ii = 0
        for ll in range(nlayers):
            if ss == 0 and ii == 0:
                no_activation = True
            else:
                no_activation = False
            if ii == 0:
                if ss == 0:
                    no_activation = True
                else:
                    no_activation = False
                in_filter = filters[ss]
                stride = self._stride_arr(strides[ss])
            else:
                in_filter = filters[ss + 1]
                stride = self._stride_arr(1)
            out_filter = filters[ss + 1]


            #保存隐藏层状态
            if ii == 0:
                self._saved_hidden.append(h)

            #建立residual块
            with tf.variable_scope("unit_{}_{}".format(ss + 1,ii)):
                h = res_func(
                    h,
                    in_filter,
                    out_filter,
                    no_activation = no_activation,
                    add_bn_ops=True
                )
                if (ii + 1) % config.num__residual_units[ss] == 0:
                    ss +=1
                    ii = 0
                else:
                    ii +=1

            #保存隐藏状态
            self._saved_hidden.append(h)


            #make a single tensor

            if type(h) == tuple:
                h = concat(h,axis=3)

            with tf.variable_scope("unit_last"):
                h = self._batch_norm("final_bn",h)
                h = self._relu("final_relu",h)
            h = self._global_avg_pool(h)#??????????

            #分类层
            with tf.variable_scope("logit"):
                logits = self._fully_connected(h,config.num_classes)
            return logits