import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。如果想要避免这种结果波动
torch.backends.cudnn.deterministic = True
# 设置为True，说明设置为使用使用非确定性算法：
torch.backends.cudnn.enabled = True
cudnn.benchmark = True
# torch.distributed 包提供分布式支持，包括 GPU 和 CPU 的分布式训练支持。
import torch.distributed as dist
# torch自带的一个优化器，里面自带了求导，更新等操作
import torch.optim as optim
from torch.utils.data import DataLoader

if __name__ == '__main__':
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda=True
