
import os
import time
import random
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from .util import save_pickle, load_pickle


class TaskData(object):

    def __init__(self):
        pass

    def train_val_split(self, X, y, valid_size, stratify=False, shuffle=True, save=True,
                        seed=None, data_name=None, data_dir=None):
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]

            for step, (data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
            del X, y
            for bt in tqdm(bucket, desc='split data'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size])
                train.extend(bt[test_size:])
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        else:
            data = []
            for step, (data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y))
            del X, y
            N = len(data)
            test_size = int(N * valid_size)
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
            valid = data[:test_size]
            train = data[test_size:]
            # 混洗train数据集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        if save:
            train_path = data_dir + "/" + data_name + ".train.pkl"
            valid_path = data_dir + "/" + data_name + ".valid.pkl"
            # valid_path = data_dir / f"{data_name}.valid.pkl"
            print("train size:{} valid size : {}" .format(len(train), len(valid)))
            save_pickle(train, train_path)
            save_pickle(valid, valid_path)

    def read_data(self, raw_data_path, preprocesor=None, is_train=True, label2id=None):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        df = pd.read_csv(raw_data_path)
        sentences = df['text'].tolist()
        if is_train:
            targets = df['label'].tolist()
            # label_list = list(set(targets))
        else:
            targets = [-1]*len(sentences)
            id = df['id']
            return id, targets, sentences

        return targets, sentences
