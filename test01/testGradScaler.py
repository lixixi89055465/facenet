# -*- coding: utf-8 -*-
# @Time : 2024/12/26 21:47
# @Author : nanji
# @Site : https://www.cnblogs.com/jimchen1218/p/14315008.html
# @File : testGradScaler.py
# @Software: PyCharm 
# @Comment :
import torch
import argparse
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int,
                        default=0)  # local_rank 制定了输出设备,
    # 默认为GPU可用列表中的第一个GPU，必须加上


#
# model, optimizer = amp.initialize(net, opt, opt_level='o1')
# net = DDP(net, delay_allreduce=True)
