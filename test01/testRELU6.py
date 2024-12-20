# -*- coding: utf-8 -*-
# @Time : 2024/12/20 21:35
# @Author : nanji
# @Site : 
# @File : testRELU6.py
# @Software: PyCharm 
# @Comment :

import torch
import torch.nn as nn

m = nn.ReLU6()
input = torch.randn(9)
output = m(input)
print('input:', input)
print('output', output)
