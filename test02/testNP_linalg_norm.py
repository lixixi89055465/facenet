# -*- coding: utf-8 -*-
# @Time : 2025/1/8 20:59
# @Author : nanji
# @Site : 
# @File : testNP_linalg_norm.py
# @Software: PyCharm 
# @Comment : https://blog.csdn.net/hqh131360239/article/details/79061535
import numpy as np

x = np.array([
    [0, 3, 4],
    [1, 6, 4]
])
print(np.linalg.norm(x))

print(np.linalg.norm(x, keepdims=True))
print('1' * 100)
print(np.linalg.norm(x, axis=1, keepdims=True))
print(np.linalg.norm(x, axis=0, keepdims=True))
print('2' * 100)
print(np.linalg.norm(x, ord=1, keepdims=True))
print(np.linalg.norm(x, ord=2, keepdims=True))
print(np.linalg.norm(x, ord=np.inf, keepdims=True))

print('3' * 100)
print(np.linalg.norm(x, ord=1, axis=1, keepdims=True))
