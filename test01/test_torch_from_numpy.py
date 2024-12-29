# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:27
# @Author : nanji
# @Site : 
# @File : test_torch_from_numpy.py
# @Software: PyCharm 
# @Comment :

import numpy
import torch

# data1 = numpy.array([5, 6, 9])
# print('data1 的数据类型为：', type(data1))
# print('data1 的值为：', data1)
#
# data2 = torch.from_numpy(data1)
# print('data2的数据类型为：', type(data2))
# print('data2的值为：', data2)
# data2[1] = 3
# print('data2的数值类型为：', type(data2))
# print('data2的值为：', data2)
# data2[1] = 3
# print('data2 的数据类型为', type(data2))
# print('data2的值为:', data2)


import numpy
import torch

data1 = numpy.array([5, 6, 9])
print('data1的数据类型为：', type(data1))
print('data1的值为：', data1)

data3 = torch.Tensor(data1)
print('data3的数据类型为：', type(data3))
print('data3的值为：', data3)