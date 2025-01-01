# -*- coding: utf-8 -*-
# @Time : 2025/1/1 15:35
# @Author : nanji
# @Site : 
# @File : testNumpy_logica.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/qq_41800366/article/details/88076180
from numpy import *

A = [True, False]
B = [False, False]
C = logical_and(A, B)
print(C)

A = arange(5)
print(A)
B = logical_and(A > 1, A < 4)
print(B)
