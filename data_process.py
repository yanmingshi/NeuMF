#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_process.py
# @Author: yanms
# @Date  : 2021/8/5 9:31
# @Desc  :


import bson
import json


def transfer2json():
    with open('./dataset/subset_iccv_board_pins.bson', 'rb') as fr, open("./dataset/subset_iccv_board_pins.json", "w",
                                                                         encoding="utf-8") as fw:
        bson_data = bson.decode_all(fr.read())
        data_list = []
        for data in bson_data:
            data['_id'] = str(data['_id'])
            data_list.append(data)
            print(data)
        json.dump(data_list, fw)


def transferpin2json():
    with open('./dataset/subset_iccv_pin_im.bson', 'rb') as f:
        pin_list = []
        bson_data = bson.decode_all(f.read())
        for data in bson_data:
            pin_list.append(data['pin_id'])

    print(len(pin_list))
    print(len(set(pin_list)))


def preprocessing_pinterest():
    pin_list = []
    board_id_list = []
    train_data = {}
    valid_data = {}
    with open('./dataset/subset_iccv_board_pins.bson', 'rb') as f:
        bson_data = bson.decode_all(f.read())
        for data in bson_data:
            pin_list.extend(data['pins'])
            board_id_list.append(data['board_id'])
    print(len(board_id_list))
    print(len(set(board_id_list)))
    pin_dict = {}
    print(len(pin_list))
    print(len(set(pin_list)))

    for index, item in enumerate(pin_list):
        pin_dict[item] = index

    # {"_id": "56c7caece4b0fd248a857cbc", "board_url": "/tressamorrison/home/", "board_id": "119134421331766936",
    #  "pins": ["119134352618674646", "119134352618662252", "119134352618656367", "119134352618650870",
    #           "119134352618446617", "119134352618440692", "119134352618427540", "119134352618423757",
    #           "119134352618416078", "119134352613427201", "119134352613427199", "119134352613427192",
    #           "119134352613427188", "119134352613427166", "119134352613427161", "119134352613126372",
    #           "119134352612827413"]}
    for index, item in enumerate(bson_data):
        train_data[index] = {}
        train_data[index]


if __name__ == '__main__':
    transferpin2json()
