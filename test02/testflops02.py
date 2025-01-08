# -*- coding: utf-8 -*-
# @Time : 2025/1/8 22:25
# @Author : nanji
# @Site : 
# @File : testflops02.py
# @Software: PyCharm 
# @Comment : https://zhuanlan.zhihu.com/p/337810633
from torchvision.models import resnet50
from thop import profile

# model = resnet50()
# flops, params = profile(model, inputs=(1, 3, 224, 224))
# print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
# print('Params= ' + str(params / 1000 ** 2) + 'M')
# 如果要使用自己的模型则:
import torch.nn as nn

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

net = models.densenet161()
macs, params = get_model_complexity_info(
    net,
    (3, 224, 224),
    as_strings=True,
    print_per_layer_stat=True,
    verbose=True
)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
