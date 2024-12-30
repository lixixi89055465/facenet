# -*- coding: utf-8 -*-
# @Time : 2024/12/30 21:45
# @Author : nanji
# @Site : 
# @File : testArgmax.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/weixin_42494287/article/details/92797061

import torch

# a = torch.tensor(
#     [
#         [1, 5, 5, 2],
#         [9, -6, 2, 8],
#         [-3, 7, -9, 1]
#     ])
# b = torch.argmax(a, dim=0)
# print(b)
# print(a.shape)
import torch

a = torch.tensor([
    [
        [1, 5, 5, 2],
        [9, -6, 2, 8],
        [-3, 7, -9, 1]
    ],

    [
        [-1, 7, -5, 2],
        [9, 6, 2, 8],
        [3, 7, 9, 1]
    ]])
b = torch.argmax(a, dim=0)
print(b)
print(a.shape)

"""
tensor([[0, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]])
torch.Size([2, 3, 4])"""

# dim=0,即将第一个维度消除，也就是将两个[3*4]矩阵只保留一个，因此要在两组中作比较，即将上下两个[3*4]的矩阵分别在对应的位置上比较

b = torch.argmax(a, dim=1)
torch.Size([2, 3, 4])
"""
# dim=1，即将第二个维度消除,这么理解：矩阵维度变为[2*4];
[1, 5, 5, 2],
[9, -6, 2, 8],
"""
b = torch.argmax(a, dim=2)
"""
tensor([[2, 0, 1],
        [1, 0, 2]])
"""
# dim=2,即将第三个维度消除，这么理解：矩阵维度变为[2*3]
"""
[1, 5, 5, 2],
[9, -6, 2, 8],
[-3, 7, -9, 1];
# 横向压缩成一维
# [2, 0, 1], 同理得到下面的"""
