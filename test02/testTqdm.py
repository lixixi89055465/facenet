# -*- coding: utf-8 -*-
# @Time : 2024/12/29 19:54
# @Author : nanji
# @Site : 
# @File : testTqdm.py
# @Software: PyCharm 
# @Comment : https://blog.csdn.net/wxd1233/article/details/118371404
import time
from tqdm import *

# for i in tqdm(range(1000)):
#     time.sleep(0.01)

# for i in trange(1000):
#     time.sleep(0.01)
# import time
# from tqdm import tqdm
#
# pbar = tqdm(["a", "b", "c", "d"])
#
# for char in pbar:
#     pbar.set_description("Processing %s" % char)  # 设置描述
#     time.sleep(1)  # 每个任务分配1s
from tqdm import tqdm
import time

for i in tqdm(range(10), desc='主要进度', position=0):
    for j in tqdm(range(100), desc='次要进度', position=1, leave=False):
        time.sleep(0.01)  # 模拟你的任务需要一些时间
