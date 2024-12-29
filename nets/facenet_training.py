# -*- coding: utf-8 -*-
# @Time : 2024/12/29 17:25
# @Author : nanji
# @Site : 
# @File : facenet_training.py
# @Software: PyCharm 
# @Comment :
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

