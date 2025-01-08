# -*- coding: utf-8 -*-
# @Time : 2025/1/8 21:48
# @Author : nanji
# @Site : 
# @File : testThop01.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/weixin_42627397/article/details/124770595
from thop import profile
from thop import clever_format
import torch
input=torch.randn(1,3,224,224)
# model=MODEL()
# ?,params=profile(model,inputs=(input,))
# ?,params=clever_format([flops,params],"%.3f")



