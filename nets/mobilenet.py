# -*- coding: utf-8 -*-
# @Time : 2024/12/19 23:09
# @Author : nanji
# @Site : 
# @File : mobilenet.py
# @Software: PyCharm 
# @Comment :
from torch import nn


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.Relu6()
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3->80,80,32
            conv_bn(3, 32, 2),
            # 80,80,32->80,80,64
            conv_dw(32, 64, 1),
            # 80,80,64->20,20,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            # 40,40,128->20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256->10,10,512

        )
