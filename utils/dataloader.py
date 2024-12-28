# -*- coding: utf-8 -*-
# @Time : 2024/12/28 17:51
# @Author : nanji
# @Site : 
# @File : dataloader.py.py
# @Software: PyCharm 
# @Comment :
import os
import random
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.dataset import Dataset
from .utils import cvtColor, preprocess_input, resize_image
import numpy as np


def rand(a=0, b=1):
    return np.random.rand() * (b - 1) + a


class FacenetDataset(Dataset):
    def __init__(self, input_shape,
                 lines,
                 num_classes,
                 random):
        self.input_shape = input_shape
        self.lines = lines
        self.length = len(lines)
        self.num_classes = num_classes
        self.random = random
        # ------------------------------------#
        #   路径和标签
        # ------------------------------------#
        self.paths = []
        self.labels = []
        self.load_dataset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # ------------------------------------#
        #   创建全为零的矩阵
        # ------------------------------------#
        images = np.zeros((3, 3, self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((3))
        # ------------------------------#
        #   先获得两张同一个人的人脸
        #   用来作为anchor和positive
        # ------------------------------#
        c = random.randint(0, self.num_classes - 1)
        selected_path = self.paths[self.labels[:] == c]
        while len(selected_path) < 2:
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]
        # ------------------------------------#
        #   随机选择两张
        # ------------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)
        # ------------------------------------#
        #   打开图片并放入矩阵
        # ------------------------------------#
        image = cvtColor(Image.open(selected_path[image_indexes[0]]))
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        if self.rand() < .5 and self.random:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
        image = preprocess_input(np.array(image, dtype='float32'))
        image = np.transpose(image, [2, 0, 1])
        images[1, :, :, :] = image
        labels[1] = c
        # ------------------------------#
        #   取出另外一个人的人脸
        # ------------------------------#
        different_c = list(range(self.num_classes))
        different_c.pop(c)
        different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
        current_c = different_c[different_c_index[0]]
        selected_path = self.paths[self.labels == current_c]
        while len(selected_path) < 1:
            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = different_c[different_c_index[0]]
            selected_path = self.paths[self.labels == current_c]
        # ------------------------------#
        #   随机选择一张
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        image = cvtColor(Image.open(selected_path[image_indexes[0]]))
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        if self.rand() < .5 and self.random:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
        image = preprocess_input(np.array(image, dtype='float32'))
        image = np.transpose(image, [2, 0, 1])
        images[2, :, :, :] = image
        labels[2] = current_c
        return images, labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(';')
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[1]))
        try:
            self.paths = np.array(self.paths, dtype=np.object)
        except:
            self.paths = np.array(self.paths, dtype=np.object_)
        self.labels = np.array(self.labels)
