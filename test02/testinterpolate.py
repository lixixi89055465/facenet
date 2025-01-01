# -*- coding: utf-8 -*-
# @Time : 2025/1/1 17:08
# @Author : nanji
# @Site :  https://blog.csdn.net/weixin_44012667/article/details/144330599
# @File : testinterpolate.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

# 创建三种不同的插值函数
f_linear = scipy.interpolate.interp1d(x, y, kind='linear')
f_quadratic = scipy.interpolate.interp1d(x, y, kind='quadratic')
f_cubic = scipy.interpolate.interp1d(x, y, kind='cubic')

# 在 xnew 处评估这些插值函数
xnew = np.linspace(0, 4, 100)
y_linear = f_linear(xnew)
y_quadratic = f_quadratic(xnew)
y_cubic = f_cubic(xnew)

# 绘制不同插值方法的结果
plt.plot(x, y, 'o', label='原始数据')
plt.plot(xnew, y_linear, '-', label='线性插值')
plt.plot(xnew, y_quadratic, '--', label='二次插值')
plt.plot(xnew, y_cubic, ':', label='三次插值')
plt.legend()
plt.show()
