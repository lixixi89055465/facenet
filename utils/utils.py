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
        dataset_path = f.readline()
    labels = []
    for path in dataset_path:
        path_split = path.split(';')
        labels.append(int(path_split))
    num_classes = np.max(labels) + 1
    return num_classes
