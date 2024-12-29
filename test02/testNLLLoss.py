# -*- coding: utf-8 -*-
# @Time : 2024/12/29 20:34
# @Author : nanji
# @Site : 
# @File : testNLLLoss.py
# @Software: PyCharm 
# @Comment : https://zhuanlan.zhihu.com/p/383044774

from torch import nn
import torch

# nllloss首先需要初始化
nllloss = nn.NLLLoss()  # 可选参数中有 reduction='mean', 'sum', 默认mean

# 在使用nllloss时，需要有两个张量，一个是预测向量，一个是label

# predict = torch.Tensor([[2, 3, 1]])  # shape: (n,category)
# label = torch.tensor([1])  # shape:(n,)


# predict=torch.Tensor([[2,3,1]])
# label=torch.tensor([1])
# print(nllloss(predict, label))
# predict = torch.Tensor([[2, 3, 1],
#                         [3, 7, 9]])
# label = torch.tensor([1, 2])
# print(nllloss(predict, label))
# nllloss = nn.NLLLoss(reduction='sum')
# predict = torch.Tensor([[2, 3, 1],
#                         [3, 7, 9]])
# label = torch.tensor([1, 2])
# nllloss(predict, label)

# nllloss = nn.NLLLoss()
# predict = torch.Tensor([
#     [2, 3, 1],
#     [3, 7, 9]
# ])
# predict = torch.log(torch.softmax(predict, dim=-1))
# label = torch.tensor([1, 2])
# print(nllloss(predict, label))
#
# cross_loss=nn.CrossEntropyLoss()
# predict = torch.Tensor([
#     [2, 3, 1],
#     [3, 7, 9]
# ])
# label=torch.tensor([1,2])
# print(cross_loss(predict, label))

cross_loss = nn.CrossEntropyLoss()
predict = torch.Tensor([
    [2, 3, 1],
    [3, 7, 9]
])
label = torch.tensor([1, 2])
print(cross_loss(predict, label))
