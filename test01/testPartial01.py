# -*- coding: utf-8 -*-
# @Time : 2024/12/29 17:02
# @Author : nanji
# @Site : 
# @File : testPartial01.py
# @Software: PyCharm 
# @Comment : https://zhuanlan.zhihu.com/p/47124891

def multiply(x, y):
    return x * y


# print(multiply(3, y=2))
# print(multiply(4, y=2))
# print(multiply(5, y=2))
def add(*args):
    return sum(args)



#
# print(add(1, 2, 3) + 100)
# print(add(5, 5, 5) + 100)


# def add(*args):
#     return sum(args) + 100
# print(add(1, 2, 3))  # 106
# print(add(5, 5, 5))  # 115
from functools import partial


def add(*args):
    return sum(args)


add_100 = partial(add, 100, 100)
print(add_100(1, 2, 3))  # 106
add_101 = partial(add, 101, 200)
print(add_101(1, 2, 3))  # 107

