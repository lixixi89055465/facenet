# -*- coding: utf-8 -*-
# @Time : 2024/12/19 21:02
# @Author : nanji
# @Site : 
# @File : utils.py
# @Software: PyCharm 
# @Comment :
import random

import numpy as np
import torch
from PIL import Image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        datapath = f.readlines()
    labels = []
    for path in datapath:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_class = np.max(labels) + 1
    return num_class


def preprocess_input(image):
    image /= 255.0
    return image


# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def show_config(**kwargs):
    print('COnfigurations:')
    print('-' * 70)
    print('|%25s | %40s |' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

