# -*- coding: utf-8 -*-
# @Time : 2025/1/1 16:54
# @Author : nanji
# @Site : 
# @File : testAxis.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/weixin_44012667/article/details/144330599

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

# 离散的已知数据点
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

f = scipy.interpolate.interp1d(x, y, kind='linear')
y_new = f(2.5)
print('插值结果:', y_new)
xnew = np.linspace(0, 4, 100)
ynew = f(xnew)
plt.plot(x, y, 'o', label='原始数据 ')
plt.plot(xnew, ynew, '-', label='线性插值 ')
plt.legend()
plt.show()
