##coding=utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)



import json
import numpy as np
import tensorflow as tf

from collections import namedtuple#返回元组数据
from resnet.models.nnlib import concat as _concat
from resnet.models.model_factory import RegisterModel
from resnet.models.resnet_model import ResNetModel
from resnet.utils import logger

log = logger.get()

@RegisterModel("revnet")
class RevNetModel(ResNetModel):
    def __init__(self,
                 config,
                 is_training=True,
                 inference_only=False,
                 inp=None,
                 label=None,
                 dtype=tf.float32,
                 batch_size=None,
                 apply_grad=True,
                 idx=0):
        if config.manmual_gradients:
            self._wd_hidden = config.wd#隐藏层衰减率
            assert self._wd_hidden > 0.0,"Not applying weight decay!"#判断隐藏层的衰减率是否大于0
            dd = config.__dict__#获取配置内容，包括key和value

            #config2是干嘛用的？？？？？？？？
            config2 = json.loads(json.dumps(config.__dict__),
                                 object_hook=lambda d:namedtuple('X',d.keys())(*d.values()))
            dd = config2.__dict__
            dd["wd"] = 0.0
            config2 = json.loads(
                json.dumps(dd),
                object_hook=lambda d:namedtuple('X',d.keys())(*d.values()))
            assert config2.wd  == 0.0,"weight decay not cleared"
            assert config.wd > 0,"weight decay not cleared"
        else:
            config2 = config
        super(RevNetModel, self).__init__(
            config2,
            is_training = is_training,
            inference_only=inference_only,
            inp = inp,
            label=label,
            batch_size=batch_size,
            apply_grad=apply_grad
        )

