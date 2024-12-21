# -*- coding: utf-8 -*-
# @Time : 2024/12/21 16:34
# @Author : nanji
# @Site : 
# @File : inception_resnetv1.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True
        )
        # //torch.sigmoid 可以使用inplace=true
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_places, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_places, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,
            momentum=0.1,
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x0, x1, x2], dim=1)
        out = self.conv2d(out)
        out = self.scale * out + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x0, x1, x2], dim=1)
        return out


class InceptionResnetV1(nn.Module):
    def __init__(self):
        super(InceptionResnetV1, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1),
        self.conv2d_3a = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(kernel_size=3, stride=2),
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1),
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1),
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2),
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
