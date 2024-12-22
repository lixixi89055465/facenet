# -*- coding: utf-8 -*-
# @Time : 2024/12/22 16:49
# @Author : nanji
# @Site : 
# @File : testsave_image01.py
# @Software: PyCharm 
# @Comment :

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision

img = plt.imread('wave.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.repeat(10, 1, 1, 1)
img_tensor = torchvision.utils.make_grid(img_tensor, nrow=4)
torchvision.utils.save_image(img_tensor, 'out.jpg')
