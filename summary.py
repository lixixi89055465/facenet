# -*- coding: utf-8 -*-
# @Time : 2025/1/6 21:11
# @Author : nanji
# @Site : 
# @File : summary.py
# @Software: PyCharm 
# @Comment :
import torch
from thop import clever_format, profile
from torchsummary import summary

if __name__ == '__main__':
    input_shape = [160, 160]
    backbone = 'mobilenet'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Facenet(num_classes=10575, backbone=backbone).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
