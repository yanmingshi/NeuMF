#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : neu_mf.py
# @Author: yanms
# @Date  : 2021/8/5 15:15
# @Desc  :
from logger import Logger

import torch
import torch.nn as nn

logging = Logger('neuMF', level='debug').logger


class GMF(nn.Module):

    def __init__(self, args):
        super(GMF, self).__init__()
        self.user_no = args.user_no
        self.item_no = args.item_no
        self.embedding_size = args.gmf_embedding_size

        self.user_embedding = nn.Embedding(self.user_no, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_no, self.embedding_size)

    def forward(self, users, items):
        users_embedding = self.user_embedding(users)
        items_embedding = self.item_embedding(items)
        scores = torch.mul(users_embedding, items_embedding)
        return scores


class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()
        self.user_no = args.user_no
        self.item_no = args.item_no
        self.embedding_size = args.mlp_embedding_size
        self.layers = args.layers  # [64,32,16,8]

        self.user_embedding = nn.Embedding(self.user_no, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_no, self.embedding_size)
        mlp_layers = []
        for _, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_layers.append(nn.Linear(in_size, out_size))
            mlp_layers.append(nn.BatchNorm1d(out_size))
            mlp_layers.append(nn.Dropout(0.25))
            mlp_layers.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        input_features = torch.cat((user_embedding, item_embedding), -1)
        return self.mlp_layers(input_features)


class NeuMF(nn.Module):

    def __init__(self, args, GMF: GMF = None, MLP: MLP = None):
        super(NeuMF, self).__init__()
        self.GMF = GMF
        self.MLP = MLP
        self.is_pretrain = args.is_pretrain
        self.sigmoid = nn.Sigmoid()
        if self.MLP is not None and self.GMF is None:
            self.neu_layer = nn.Linear(self.MLP.layers[-1], 1)
        elif self.GMF is not None and self.MLP is None:
            self.neu_layer = nn.Linear(self.GMF.embedding_size, 1)
        elif self.GMF is not None and self.MLP is not None:
            self.neu_layer = nn.Linear(self.MLP.layers[-1] + self.GMF.embedding_size, 1)
        else:
            raise Exception("GMF and MLP can't both None")
        self.gmf_ckpt_path = args.gmf_ckpt_path
        self.mlp_ckpt_path = args.mlp_ckpt_path
        self.full_ckpt_path = args.full_ckpt_path

        if self.is_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            module.bias.data.zero_()

    def load_pretrain(self):
        if self.gmf_ckpt_path is not None and self.mlp_ckpt_path is not None:
            gmf_ckpt = torch.load(self.gmf_ckpt_path)
            mlp_ckpt = torch.load(self.mlp_ckpt_path)
            self.GMF.load_state_dict(gmf_ckpt['GMF'])
            self.MLP.load_state_dict(mlp_ckpt['MLP'])
            gmf_neu_layer_w = gmf_ckpt['neu_layer']
            mlp_neu_layer_w = gmf_ckpt['neu_layer']
            self.neu_layer.weight.data = torch.cat((gmf_neu_layer_w['weight']*0.5, mlp_neu_layer_w['weight']*0.5), -1)
            self.neu_layer.bias.data = (gmf_neu_layer_w['bias'] + mlp_neu_layer_w['bias']) * 0.5

        elif self.full_ckpt_path is not None:
            params = torch.load(self.full_ckpt_path)
            self.load_state_dict(params)
        elif self.gmf_ckpt_path is not None:
            gmf_ckpt = torch.load(self.gmf_ckpt_path)
            self.GMF.load_state_dict(gmf_ckpt['GMF'])
            self.neu_layer.load_state_dict(gmf_ckpt['neu_layer'])
        elif self.mlp_ckpt_path is not None:
            mlp_ckpt = torch.load(self.mlp_ckpt_path)
            self.MLP.load_state_dict(mlp_ckpt['MLP'])
            self.neu_layer.load_state_dict(mlp_ckpt['neu_layer'])

        logging.info('loading checkpoint success!')

    def forward(self, users, items):
        if self.MLP is not None and self.GMF is None:
            mlp_score = self.MLP(users, items)
            scores = self.neu_layer(mlp_score)
        elif self.GMF is not None and self.MLP is None:
            gmf_score = self.GMF(users, items)
            scores = self.neu_layer(gmf_score)
        elif self.GMF is not None and self.MLP is not None:
            mlp_score = self.MLP(users, items)
            gmf_score = self.GMF(users, items)
            scores = self.neu_layer(torch.cat((gmf_score, mlp_score), dim=-1))
        else:
            raise Exception("GMF and MLP can't both None")
        scores = scores.view(-1)
        return self.sigmoid(scores)
