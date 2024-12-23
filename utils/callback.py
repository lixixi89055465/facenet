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
        dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
        self.writer.add_graph(model, dummy_input)

    def append_loss(self, epoch, acc, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.acc.append(acc)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, 'epoch_acc.txt'), 'a') as f:
            f.write(str(acc))
            f.write('\n')
        with open(os.path.join(self.log_dir, 'epoch_loss.txt'), 'a') as f:
            f.write(str(loss))
            f.write('\n')
        with open(os.path.join(self.log_dir, 'epoch_val_loss.txt'), 'a') as f:
            f.write(str(val_loss))
            f.write('\n')
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()
