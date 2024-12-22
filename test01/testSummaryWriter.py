# -*- coding: utf-8 -*-
# @Time : 2024/12/22 13:53
# @Author : nanji
# @Site : 
# @File : testSummaryWriter.py
# @Software: PyCharm 
# @Comment :

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/ZCH_Tensorboard_Trying_logs')
for i in range(100):
    writer.add_scalar("y=x", i, i)
for i in range(100):
    writer.add_scalar('y=2*x', 2 * i, i)

writer.close()