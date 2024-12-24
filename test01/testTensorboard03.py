# -*- coding: utf-8 -*-
# @Time : 2024/12/24 22:24
# @Author : nanji
# @Site : 
# @File : testTensorboard03.py
# @Software: PyCharm 
# @Comment :https://www.cnblogs.com/sddai/p/14516691.html
#3.4 Tensorboard综合Demo
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

resnet18 = models.resnet18(False)
writer = SummaryWriter()
sample_rate = 44100
