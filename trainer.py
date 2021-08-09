#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author: yanms
# @Date  : 2021/8/5 15:43
# @Desc  :
import argparse
import copy
import random
import time

import torch
import numpy as np
from torch import optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from logger import Logger
from data_set import DataSet, generate_negative_train_data
from metrics import metrics_dict
from neu_mf import GMF, MLP, NeuMF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging = Logger('trainner', level='debug').logger

CHKPOINT_PATH = "./check"

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数
torch.cuda.manual_seed(SEED)  # 为GPU设置种子用于生成随机数
torch.cuda.manual_seed_all(SEED)  # 为多个GPU设置种子用于生成随机数


class Trainer(object):

    def __init__(self, model, args):
        self.loss = BCELoss()
        self.model = model.to(device)
        self.signal = args.signal
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.negative_count = args.negative_count
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.topk = args.topk
        self.metrics = args.metrics
        self.min_epoch = args.min_epoch
        self.optimizer = self.get_optimizer(self.model)
        self.writer = SummaryWriter('./log/' + args.model_name + '-' + str(time.time()))

    def get_loss(self, scores, labels):
        return self.loss(scores, labels)

    def get_optimizer(self, model):
        params = [{'params': value} for _, value in model.named_parameters()]
        if self.signal:
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def save_model(self, best_model):
        # 保存模型
        if self.model.MLP is not None and self.model.GMF is None:
            torch.save({'MLP': best_model.MLP.state_dict(), 'neu_layer': best_model.neu_layer.state_dict()},
                       CHKPOINT_PATH + '/MLP_b' + str(self.batch_size) + '_emb_' + str(self.model.MLP.embedding_size) + '_neg_' + str(self.negative_count) + '.pth')
        elif self.model.GMF is not None and self.model.MLP is None:
            torch.save({'GMF': best_model.GMF.state_dict(), 'neu_layer': best_model.neu_layer.state_dict()},
                       CHKPOINT_PATH + '/GMF_b' + str(self.batch_size) + '_emb_' + str(self.model.GMF.embedding_size) + '_neg_' + str(self.negative_count) + '.pth')
        elif self.model.GMF is not None and self.model.MLP is not None:
            torch.save(best_model, CHKPOINT_PATH + '/newMF_b' + str(self.batch_size) + '_emb_' + str(self.model.GMF.embedding_size) + '_neg_' + str(self.negative_count) + '.pth')
        else:
            raise Exception("GMF and MLP can't both None")

    @torch.no_grad()
    def evaluate(self, data_loader, epoch):
        self.model.eval()
        start_time = time.time()

        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(device)
            items = batch_data[:, 1:]
            users = batch_data[:, 0].view(-1, 1).repeat(1, items.shape[1])
            scores = self.model(users.reshape(-1), items.reshape(-1))
            scores = scores.view(items.shape)
            _, topk_idx = torch.topk(scores, self.topk, dim=-1)
            topk = []
            # 根据topk的下标获取topk的item编号
            for i, item in enumerate(topk_idx):
                topk.append(items[i, item].unsqueeze(0))
            topks = torch.cat(topk, 0)
            # 拿到gt值，然后用topk的item值与gt值做减法 等于0则为命中的预测
            gt = items[:, 0]
            gts = gt.view(-1, 1).repeat(1, self.topk)
            mask = topks - gts == 0
            topk_list.extend(mask.cpu())
        topk_list = torch.cat(topk_list).view(-1, 10).numpy()
        gt_len = np.ones(len(topk_list), dtype=int)
        metric_dict = self.calculate_result(topk_list, gt_len, epoch)
        epoch_time = time.time() - start_time
        logging.info(f"evaluator %d cost time %.2fs, result: %s " % (epoch, epoch_time, metric_dict.__str__()))
        return metric_dict

    def calculate_result(self, topk_list, gt_len, epoch):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for metric, value in zip(self.metrics, result_list):
            key = '{}@{}'.format(metric, self.topk)
            metric_dict[key] = np.round(value[self.topk - 1], 4)
            self.writer.add_scalar('evaluate ' + metric, metric_dict[key], epoch + 1)
        return metric_dict

    def train_model(self, train_dataset, validate_dataset):

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        validate_loader = DataLoader(dataset=validate_dataset, batch_size=self.batch_size)
        min_loss = 10  # 用来保存最好的模型
        best_model = None
        best_hit, best_ndcg, best_epoch = 0.0, 0.0, 0
        for epoch in range(self.epochs):
            total_loss = 0.0
            self.model.train()
            start_time = time.time()
            train_data_iter = (
                tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"\033[1;35mTrain {epoch:>5}\033[0m"
                )
            )
            batch_no = 0
            for batch_index, batch_data in train_data_iter:
                batch_data = generate_negative_train_data(batch_data, train_dataset.item_count,
                                                          self.negative_count)
                batch_data = batch_data.to(device)
                self.optimizer.zero_grad()
                scores = self.model(batch_data[:, 0], batch_data[:, 1])
                loss = self.get_loss(scores, batch_data[:, -1].float())
                loss.backward()
                self.optimizer.step()

                batch_no = batch_index
                total_loss += loss

                # 记录loss到tensorboard可视化
                self.writer.add_scalar('training loss', loss, epoch * self.batch_size + len(batch_data))
                # if batch_index % 10 == 9:
                #     logging.info(
                #         'epoch %d minibatch %d train loss is [%.4f] ' % (epoch + 1, batch_index + 1, total_loss))
            total_loss = total_loss / (batch_no + 1)
            epoch_time = time.time() - start_time
            logging.info('epoch %d %.2fs train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

            # writer.add_scalar('training loss', total_loss, (epoch + 1) * len(train_loader))

            # 保存最好的模型
            # if epoch > self.min_epoch and total_loss <= min_loss:
            #     min_loss = total_loss
            #     best_model = copy.deepcopy(self.model)

            metric_dict = self.evaluate(validate_loader, epoch)
            hit, ndcg = metric_dict['hit@' + str(self.topk)], metric_dict['ndcg@' + str(self.topk)]
            if epoch > self.min_epoch and hit > best_hit:
                best_hit, best_ndcg, best_epoch = hit, ndcg, epoch
                best_model = copy.deepcopy(self.model)
                # 保存最好的模型
                self.save_model(best_model)

        logging.info(f"training end, best iteration %d, results: hit@{self.topk}: %s, ndgc@{self.topk}: %s" %
                     (best_epoch+1, best_hit, best_ndcg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--gmf_embedding_size', type=str, default=8, help='')
    parser.add_argument('--mlp_embedding_size', type=str, default=32, help='')
    parser.add_argument('--layers', type=str, default=[64, 32, 16, 8], help='')
    parser.add_argument('--lr', type=str, default=0.001, help='')
    parser.add_argument('--decay', type=str, default=1e-4, help='')
    parser.add_argument('--negative_count', type=str, default=4, help='')
    parser.add_argument('--batch_size', type=str, default=512, help='')
    parser.add_argument('--epochs', type=str, default=50, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--topk', type=str, default=10, help='')
    parser.add_argument('--metrics', type=str, default=['hit', 'ndcg'], help='')
    parser.add_argument('--gmf_ckpt_path', type=str, default='./check/GMF.pth', help='')
    parser.add_argument('--mlp_ckpt_path', type=str, default='./check/MLP.pth', help='')
    parser.add_argument('--full_ckpt_path', type=str, default=None, help='')

    parser.add_argument('--signal', type=str, default=True, help='')
    parser.add_argument('--is_pretrain', type=str, default=False, help='')
    parser.add_argument('--model_name', type=str, default='neuMF', help='')
    parsers = argparse.ArgumentParser('training and evaluation script', parents=[parser])
    args = parsers.parse_args()

    train_file_name = './dataset/ml-1m.train.rating'
    validate_file_name = './dataset/ml-1m.test.negative'
    train_dataset = DataSet(train_file_name, type='train')
    validate_dataset = DataSet(validate_file_name, type='validate')

    args.user_no = train_dataset.user_count
    args.item_no = train_dataset.item_count

    gmf = GMF(args)
    mlp = MLP(args)
    model = NeuMF(args, gmf, None)
    # model = NeuMF(args, None, mlp)
    # model = NeuMF(args, gmf, mlp)
    logging.info(model)
    trainer = Trainer(model, args)
    start = time.time()
    trainer.train_model(train_dataset, validate_dataset)
    # validate_dataloader = DataLoader(validate_dataset, 10)
    # trainer.evaluate(validate_dataloader, 0)
    logging.info("training end total use time :%.2fs" % (time.time() - start))
