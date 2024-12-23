# -*- coding: utf-8 -*-
# @Time : 2024/12/23 21:45
# @Author : nanji
# @Site : 
# @File : testTensorboard01.py
# @Software: PyCharm 
# @Comment :https://www.cnblogs.com/sddai/p/14516691.html
#pip install tensorboardX、pip install tensorflow
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./data/to/log')
# writer.add_scalar('acc',scalar_value,global_step=None,walltime=None)
# tag = 'acc'
# for epoch in range(100):
#     mAP=eval(model)
    # writer.add_scalar('mAP', epoch * 2, epoch)
#
# writer.close()
import numpy as np
from tensorboardX import SummaryWriter
writer=SummaryWriter()
for epoch in range(100):
    writer.add_scalar('scalar/test',np.random.rand(),epoch)
    writer.add_scalar('scalar/scalar_test',{'xsinx':epoch*np.sin(epoch),'xcosx':epoch*np.cos(epoch)})

