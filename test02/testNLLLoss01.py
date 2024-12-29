# -*- coding: utf-8 -*-
# @Time : 2024/12/29 22:36
# @Author : nanji
# @Site : 
# @File : testNLLLoss01.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/watermelon1123/article/details/91044856

import torch
import torch.nn as nn

a = torch.Tensor([1, 2, 3])
# 定义Softmax
softmax = nn.Softmax()
sm_a = softmax = nn.Softmax()
print(sm_a)
# 输出：tensor([0.0900, 0.2447, 0.6652])

# 定义LogSoftmax
# logsoftmax = nn.LogSoftmax()
# lsm_a = logsoftmax(a)
# print(lsm_a)
# 输出tensor([-2.4076, -1.4076, -0.4076])，其中ln(0.0900)=-2.4076
