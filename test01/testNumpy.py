# -*- coding: utf-8 -*-
# @Time : 2024/12/28 22:37
# @Author : nanji
# @Site : 
# @File : testNumpy.py
# @Software: PyCharm 
# @Comment :https://geek-docs.com/numpy/numpy-ask-answer/numpy-random-choice_z1.html


import numpy as np

# 生成随机矩阵
# matrix = np.random.choice(['X', 'O', 'numpyarray.com'], size=(3, 3))
# print(matrix)
# arr = np.array(['A', 'B', 'C', 'D', 'numpyarray.com'])
# shuffeld = np.random.choice(arr, size=len(arr), replace=False)
# print(shuffeld)

# 生成随机字符串
# chars = np.array(list('abcdefghijklmnopqrstuvwxyzNUMPYARRAY.COM'))
# random_string = ''.join(np.random.choice(chars, size=10))
# print(random_string)

import numpy as np

#
# data = np.array([1, 2, 3, 4, 5, 5, 6, 6, 78, 9, 'numpyarray.com'])
# sample = np.random.choice(data, size=5, replace=False)
# print(sample)

# coin = np.array(['heads', 'tails', 'numpyarray.com'])
# results = np.random.choice(coin, size=10, p=[0.45, 0.45, 0.1])
# print(results)
# 加权随机选择
# items = np.array(['iterm1', 'item2', 'item3', 'numparray.com'])
# weights = np.array([10, 20, 30, 5])
# normalized_weights = weights / np.sum(weights)
# choice = np.random.choice(items, size=1, p=normalized_weights)
# print(choice)


arr = np.array(['a', 'b', 'c', 'd', 'e', 'numpyarray.com'])
random_indices = np.random.choice(len(arr), size=3, replace=False)
random_elements = arr[random_indices]
print(random_elements)
