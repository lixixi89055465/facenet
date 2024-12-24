# -*- coding: utf-8 -*-
# @Time : 2024/12/24 21:47
# @Author : nanji
# @Site : 
# @File : testTensorboard02.py
# @Software: PyCharm 
# @Comment :https://www.cnblogs.com/sddai/p/14516691.html

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.01
writer = SummaryWriter(comment='Linear')
x_train = np.array([
    [3.3], [4.4], [5.5], [6.7], [6.9], [4.1],
    [9.7], [6.1], [7.5], [2.1], [7.0], [10.7],
    [5.3], [7.9], [3.1]], dtype=np.float32)
y_train = np.array([
    [1.7], [2.7], [2.0], [3.1], [1.6], [1.5],
    [3.3], [2.5], [2.5], [1.2], [2.8], [3.4],
    [1.6], [2.9], [1.3]], dtype=np.float32)
model = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    writer.add_scalar('Train', loss, epoch)
    if (epoch + 1) % 5 == 0:
        print('Epoch {}/{} , loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
writer.add_graph(model, (inputs,))
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
writer.close()
