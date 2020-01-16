# coding: UTF-8
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm  # 进度条工具
import time
from datetime import timedelta
import pandas as pd
from transformers import BertTokenizer

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD, CLS = '<UNK>', '<PAD>', '[CLS]'  # 未知字，padding符号


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def build_dataset_bert(config, args):
    def load_dataset(path, pad_size=180):
        contents = []
        lines = load_pickle(path)
        tokenizer = BertTokenizer.from_pretrained(config['bert_model_dir'])
        print('bert model load train data:{}\n'.format(path))
        for line in tqdm(lines):
            if not line:
                continue
            content, label = line[0], line[1]
            token = tokenizer.tokenize(content)

            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * \
                        (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
            contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config['train_train_path'],
                         pad_size=args.train_max_seq_len)
    valid = load_dataset(config['train_valid_path'],
                         pad_size=args.train_max_seq_len)
    # test = load_dataset(config.test_path, config.pad_size)
    return train, valid


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        # self.model_name = model_name

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index *
                                   self.batch_size:len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index *
                                   self.batch_size:(self.index + 1) *
                                   self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(config, args):
    train_data, valid_data = build_dataset_bert(config, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter = DatasetIterater(train_data, args.train_batch_size, device)
    valid_iter = DatasetIterater(valid_data, args.train_batch_size, device)
    return train_iter, valid_iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
