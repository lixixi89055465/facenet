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
from nets.mobilenet import MobileNetV1
from nets.inception_resnetv1 import InceptionResnetV1


class mobilenet(nn.Module):
    def __init__(self, backbone='mobilenet',
                 dropout_keep_prob=0.5,
                 embedding_size=128, num_classes=None,
                 mode='train', pretrained=False):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth",
                model_dir='model_data',
                progress=True)
            self.model.load_state_dict(state_dict)
        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth",
                model_dir='model_data',
                progress=True
            )

    pass


class Facenet(nn.Module):
    def __init__(self, backbone='mobilenet',
                 dropout_keep_prob=0.5,
                 embedding_size=128, num_classes=None,
                 mode='train', pretrained=False):
        super(Facenet, self).__init__()
        if backbone == 'mobilenet':
            self.backbone = mobilenet(pretrained)
            flat_shape = 1024
        elif backbone == 'inception_resnetv1':
            self.backbone = inception_resnet(pretrained)

        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth",
                model_dir='model_data',
                progress=True)
            self.model.load_state_dict(state_dict)
        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x
