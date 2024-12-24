# -*- coding: utf-8 -*-
# @Time : 2024/12/24 22:24
# @Author : nanji
# @Site : 
# @File : testTensorboard03.py
# @Software: PyCharm 
# @Comment :https://www.cnblogs.com/sddai/p/14516691.html
# 3.4 Tensorboard综合Demo
import torch
# import torchvision.utils as utils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

resnet18 = models.resnet18(False)
writer = SummaryWriter()
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]
for n_iter in range(100):
    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    # data grouping by 'slash'
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)
    writer.add_scalar('data/scalar_group', {
        'xsinx': n_iter * np.sin(n_iter),
        'xcosx': n_iter * np.cos(n_iter),
        'arctanx': np.arctan(n_iter)
    }, n_iter)
    writer.add_scalar('data/scalar_group', {
        'xsinx': n_iter * np.sin(n_iter),
        'xcosx': n_iter * np.cos(n_iter),
        'arctanx': np.arctan(n_iter)
    }, n_iter)
    dummy_img = torch.rand(32, 3, 64, 64)
    if n_iter % 10 == 0:
        x = torch.utils.make_grid(dummy_img, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)
        dummy_audio = torch.zeros(sample_rate * 2)
