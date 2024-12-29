# -*- coding: utf-8 -*-
# @Time : 2024/12/29 17:18
# @Author : nanji
# @Site : 
# @File : testTorchRand.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/qq_42951560/article/details/112174334

import torch

# torch.manual_seed(0)
# print(torch.rand(1))

# print(torch.rand(1)) # 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数

# test.py
# import torch
# torch.manual_seed(0)
# print(torch.rand(1))
# print(torch.rand(1))

# test.py
import torch
torch.manual_seed(0)
print(torch.rand(1))
torch.manual_seed(0)
print(torch.rand(1))