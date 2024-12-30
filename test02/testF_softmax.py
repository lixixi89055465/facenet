# -*- coding: utf-8 -*-
# @Time : 2024/12/30 21:23
# @Author : nanji
# @Site : 
# @File : testF_softmax.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/qq_43359515/article/details/126083252

import torch
import torch.nn.functional as F

# input = torch.randn(3, 4)
# print('input=', input)
# b = F.softmax(input, dim=0)
# print(b)
#
# print('0000000000000')
# C = F.softmax(input, dim=1)
# print(C)

import torch
import torch.nn.functional as F

a = torch.rand(3, 4, 5)
print(a)
b=F.softmax(a,dim=0)
print(b)
