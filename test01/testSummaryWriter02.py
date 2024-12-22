# -*- coding: utf-8 -*-
# @Time : 2024/12/22 14:01
# @Author : nanji
# @Site : 
# @File : testSummaryWriter02.py
# @Software: PyCharm 
# @Comment :https://pytorch.ac.cn/docs/stable/tensorboard.html

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5), )
])
trainset = torchvision.datasets.MNIST('mnist_train', train=True,
                                      download=True,  #
                                      transform=transform)
trainloader = torchvision.datasets.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7,  #
                              stride=2,  #
                              padding=3, bias=False)
images, labels = next(iter(trainloader))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
