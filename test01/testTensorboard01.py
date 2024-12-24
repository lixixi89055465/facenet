# -*- coding: utf-8 -*-
# @Time : 2024/12/23 21:45
# @Author : nanji
# @Site : 
# @File : testTensorboard01.py
# @Software: PyCharm 
# @Comment :https://www.cnblogs.com/sddai/p/14516691.html
# pip install tensorboardX、pip install tensorflow
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./data/to/log')
# writer.add_scalar('acc',scalar_value,global_step=None,walltime=None)
# tag = 'acc'
# for epoch in range(100):
#     mAP=eval(model)
# writer.add_scalar('mAP', epoch * 2, epoch)
#
# writer.close()
import numpy as np
from tensorboardX import SummaryWriter

# writer = SummaryWriter()
# for epoch in range(100):
#     writer.add_scalar('scalar/test', np.random.rand(), epoch)
#     writer.add_scalar('scalar/scalar_test',
#                       {'xsinx': epoch * np.sin(epoch), 'xcosx': epoch * np.cos(epoch)})
# writer.close()

import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.nn as nn


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


import torch

dummy_input = torch.randn(13, 1, 28, 28)
model = Net1()
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model, (dummy_input,))
