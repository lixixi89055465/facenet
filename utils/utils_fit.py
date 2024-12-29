# -*- coding: utf-8 -*-
# @Time : 2024/12/29 17:34
# @Author : nanji
# @Site : 
# @File : utils_fit.py
# @Software: PyCharm 
# @Comment :

def fit_one_epoch(model_train, model, loss_history, loss, optimizer,
                  epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, test_loader, Batch_size,
                  lfw_eval_flag, fp16, scaler,
                  save_period, save_dir, local_rank):
    total_triple_loss = 0
    total_CE_loss = 0

