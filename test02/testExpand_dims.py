# -*- coding: utf-8 -*-
# @Time : 2025/1/7 22:34
# @Author : nanji
# @Site : 
# @File : testExpand_dims.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/qq_37924224/article/details/119816771


import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
b = np.expand_dims(
    a,
    axis=0
)


print(b)
b=np.expand_dims(a,axis=1)
print(b)