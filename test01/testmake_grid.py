# -*- coding: utf-8 -*-
# @Time : 2024/12/22 17:27
# @Author : nanji
# @Site : 
# @File : testmake_grid.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/qq7835144/article/details/108523997
from torchvision.utils import save_image, make_grid
import numpy as np
import torch
import matplotlib.pyplot as plt


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


images = torch.FloatTensor(
    100 * np.random.normal(0, 1, (25, 1, 28, 28)))
show(make_grid(images, nrow=5, padding=10, pad_value=0))
plt.show()
