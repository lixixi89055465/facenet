# -*- coding: utf-8 -*-
# @Time : 2024/12/26 21:47
# @Author : nanji
# @Site : https://www.cnblogs.com/jimchen1218/p/14315008.html
# @File : testGradScaler.py
# @Software: PyCharm 
# @Comment :
import torch

tensor1 = torch.zeros(30, 20)
print(tensor1.type())
tensor2 = torch.Tensor([1, 2])
print(tensor2)

from apex import amp
