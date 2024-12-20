# -*- coding: utf-8 -*-
# @Time : 2024/12/19 21:02
# @Author : nanji
# @Site : 
# @File : utils.py
# @Software: PyCharm 
# @Comment :
import numpy as np


def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        datapath = f.readlines()
    labels = []
    for path in datapath:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_class = np.max(labels) + 1
    return num_class
