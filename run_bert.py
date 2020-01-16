# coding: UTF-8

import time
import numpy as np
import argparse
from argparse import ArgumentParser
from importlib import import_module

import torch
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup

from tools.util import build_dataset_bert, build_iterator, get_time_dif
from train_eval import bert_train

import warnings
warnings.filterwarnings("ignore")

config = {
    'raw_data_path': 'dataset/train.csv',
    'test_path': 'dataset/test_stage1.csv',
    'train_train_path': 'dataset/train.train.pkl',
    'train_valid_path': 'dataset/train.valid.pkl',

    'data_dir': 'dataset',
    'log_dir': 'output/log',
    'save_path': 'output/save',

    'bert_vocab_path': 'pretrain/bert/bert-base-chinese/vocab.txt',
    'bert_config_file': 'pretrain/bert/bert-base-chinese/config.json',
    'bert_model_dir': 'pretrain/bert/bert-base-chinese',

    'xlnet_model_dir': 'pretrain/xlnet',
    'roberta_model_dir': 'pretrain/roberta',
    'ernie_model_dir': 'pretrain/ernie/ERNIE_stable-1.0.1',
    'albert_model_dir': 'pretrain/albert',
    'bert_wwm_model_dir': 'pretrain/bert_wwm',

    'hidden_size': 768,
    'class_list': ['0', '1'],
    'num_classes': 2,
    'filter_sizes': [2, 3, 4],  # textCNN参数
    'num_filters': 256,  # textCNN参数
    'dropout': 0.1,  # textCNN参数
}


def run_train(args, config):
    '''
    param: 
    return: 
    '''
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    #  加载数据
    start_time = time.time()
    print("start loading data...\n")
    train_iter, valid_iter = build_iterator(config, args)
    time_dif = get_time_dif(start_time)
    print("\nload data time usage : ", time_dif)
    # # 开始训练
    start_time = time.time()
    if args.model_name == "ernie":
        config['bert_model_dir'] = config['ernie_model_dir']
    elif args.model_name == "xlnet":
        config['bert_model_dir'] = config['xlnet_model_dir']
    elif args.model_name == "roberta":
        config['bert_model_dir'] = config['roberta_model_dir']
    elif args.model_name == "albert":
        config['bert_model_dir'] = config['albert_model_dir']
    elif args.model_name == "bert_wwm":
        config['bert_model_dir'] = config['bert_wwm_model_dir']

    from models.Bert import Model
    model = Model(config)
    if args.use_cnn:
        from models.Bert_CNN import Model as bert_cnn
        model = bert_cnn(config)

    print(model.parameters)
    print("------\ntraining: model_name: ", args.model_name)

    bert_train(config, args,  model, train_iter, valid_iter)

    time_dif = get_time_dif(start_time)
    print("\ntrain time usage: ", time_dif)


def main():
    '''
    函数入口
    '''
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='train', type=str)
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--mode", default='max', type=str)
    parser.add_argument("--monitor", default='valid_f1', type=str)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int,
                        help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0',
                        help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=int, )
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--require_improvement', default=1000, type=int)
    parser.add_argument('--model_name', default='bert', type=str,
                        help=" bert、ernie、xlnet、roberta、albert")
    parser.add_argument('--use_cnn', default=0, type=int, 
        help="0: bert base, 1: bert+cnn")
    args = parser.parse_args()
    if args.do_data:
        from tools.task_data import TaskData
        data = TaskData()
        targets, sentences = data.read_data(
            raw_data_path=config['raw_data_path'])
        data.train_val_split(X=sentences, y=targets, shuffle=True, stratify=targets,
                             valid_size=args.valid_size,  data_dir=config['data_dir'], data_name=args.data_name)

    if args.do_train:
       run_train(args, config)

    if args.do_test:
        pass


if __name__ == '__main__':
    main()
