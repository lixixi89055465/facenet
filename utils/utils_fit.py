# -*- coding: utf-8 -*-
# @Time : 2024/12/29 17:34
# @Author : nanji
# @Site : 
# @File : utils_fit.py
# @Software: PyCharm
# @Comment :
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_one_epoch(model_train, model, loss_history, loss, optimizer,
                  epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, test_loader, Batch_size,
                  lfw_eval_flag, fp16, scaler,
                  save_period, save_dir, local_rank):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,
                    desc=f'Epoch {epoch + 1}/{epoch}',
                    postfix=dict,
                    mininterval=0.3)
        model_train.train()
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images = images.cuda(local_rank)
                    labels = labels.cuda(local_rank)
            optimizer.zero_grad()
            if not fp16:
                outputs1, outputs2 = model_train(images, 'train')
                _triple_loss = loss(outputs1, Batch_size)
                _CE_loss = nn.NLLLoss()(F.log_sof)
                _loss = _triple_loss + _CE_loss

                _loss.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs1, outputs2 = model_train(images, 'train')
                    _triple_loss = loss(outputs1, Batch_size)
                    _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                    _loss = _triple_loss + _CE_loss
                # ----------------------#
                #   反向传播
                # ----------------------#
                scaler.scale(_loss).backward()
                scaler.step(optimizer)
                scaler.update()
