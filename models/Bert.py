# coding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        print(config['bert_model_dir'])
        self.bert = BertModel.from_pretrained(config['bert_model_dir'])

        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config['hidden_size'], config['num_classes'])

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        return out
