#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author: yanms
# @Date  : 2021/8/5 16:46
# @Desc  :
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class DataSet(Dataset):

    def __init__(self, file_name, type: str):
        self.user_count = 0
        self.item_count = 0
        self.data_size = 0
        self.data = None
        self.type = type
        self.init(file_name)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data_size

    def init(self, file_name):

        if self.type.lower() == 'train':
            df = pd.read_csv(file_name, usecols=[0, 1], sep="\t", header=None)
            self.user_count = df[0].max() + 1
            self.item_count = df[1].max() + 1

        else:
            df = pd.read_csv(file_name, sep='\t', header=None)
            self.data_size = len(df)
            df1 = df[0].str[1:-1].str.split(',', expand=True)
            df = df.drop(0, axis=1)
            df.insert(0, 'gt', df1[1].astype("int"))
            df.insert(0, 'user', df1[0].astype("int"))
        self.data_size = len(df)
        self.data = df.to_numpy()


def generate_negative_train_data(data, item_count, negative_count):
    """
    返回包含负样本的训练数据
    :param data:
    :param item_count:
    :param negative_count:
    :return:
    """
    data_set = []
    for pairs in data:
        user, positive = pairs[0].item(), pairs[1].item()
        data_set.append([user, positive, 1])
        for _ in range(negative_count):
            negative = np.random.randint(item_count)
            while [user, negative] in data_set:
                negative = np.random.randint(item_count)
            data_set.append([user, negative, 0])
    data_set = torch.IntTensor(data_set)
    return data_set


if __name__ == '__main__':
    train_file_name = './dataset/ml-1m.train.rating'
    validate_file_name = './dataset/ml-1m.test.negative'
    train_dataset = DataSet(train_file_name, type='train')
    validate_dataset = DataSet(validate_file_name, type='validate')
    data_loader = DataLoader(dataset=train_dataset, batch_size=3, shuffle=True)
    for index, item in enumerate(data_loader):
        result = generate_negative_train_data((index, item), train_dataset.item_count, 3)
        print('index:', index)
        print('item:', item)
        result = result[1]
        print(result[:,-1])
        break
    # generate_negative_train_data(i)
    valid_data_loader = DataLoader(dataset=validate_dataset, batch_size=3)
    for item in enumerate(valid_data_loader):
        a, b = item
        print(a)
        print(b)
        break
