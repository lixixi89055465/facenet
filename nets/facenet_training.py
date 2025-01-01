# -*- coding: utf-8 -*-
# @Time : 2024/12/29 17:25
# @Author : nanji
# @Site : 
# @File : facenet_training.py
# @Software: PyCharm 
# @Comment :
import math


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr_scheduler(lr_decay_type,
                     lr,
                     min_lr,
                     total_iters,
                     warmup_iters_ratio=0.1,
                     warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3,
                     step_num=10):
    def yolox_warm_cos_lr(
            lr,
            min_lr,
            total_iters,
            warmup_total_iters,
            warmup_lr_start,
            no_aug_iter,
            iters):
        if iters <= warmup_total_iters:
            lr = ((lr - warmup_lr_start) *
                  pow(iters / float(warmup_total_iters), 2) +
                  warmup_lr_start)
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    +
                    math.cos(
                        math.pi * (iters - warmup_total_iters)
                        / (total_iters - warmup_total_iters - no_aug_iter)
                    )
            )
        return lr
