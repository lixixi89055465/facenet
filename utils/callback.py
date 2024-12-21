# -*- coding: utf-8 -*-
# @Time : 2024/12/21 21:59
# @Author : nanji
# @Site : 
# @File : callback.py
# @Software: PyCharm 
# @Comment :

import datetime
import os
import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        self.log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        self.acc = []
        self.losses = []
        self.val_loss = []
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
