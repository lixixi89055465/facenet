# -*- coding: utf-8 -*-
# @Time : 2024/12/19 22:14
# @Author : nanji
# @Site : 
# @File : facenet.py
# @Software: PyCharm 
# @Comment :
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from nets.mobilenet import MobilenetV1


class Facenet(nn.Module):
    def __init__(self, backbone='mobilenet',
                 dropout_keep_prob=0.5,
                 embedding_size=128, num_classes=None,
                 mode='train', pretrained=False):
        super(Facenet,self).__init__()
        if backbone=='mobilenet':
            self.backbone=mobilenet()
