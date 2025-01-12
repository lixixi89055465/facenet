# -*- coding: utf-8 -*-
# @Time : 2025/1/12 16:11
# @Author : nanji
# @Site : 
# @File : testSummaryWriter01.py
# @Software: PyCharm 
# @Comment : https://blog.csdn.net/weixin_46254985/article/details/136332066
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()
