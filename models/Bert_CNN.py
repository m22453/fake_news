# coding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer

# class Model(nn.Module):


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_model_dir'])

        for param in self.bert.parameters():
            param.requires_grad = True

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config['num_filters'], (k, config['hidden_size']))
             for k in config['filter_sizes']]
        )
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(config['num_filters'] *
                            len(config['filter_sizes']), config['num_classes'])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]
        mask = x[2]

        encoder_out, text_cls = self.bert(
            context, attention_mask=mask)

        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
