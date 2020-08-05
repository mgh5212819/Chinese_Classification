'''
@Author: https://github.com/649453932/
@Date: 2020-04-09 15:59:02
@LastEditTime: 2020-06-30 14:12:53
@LastEditors: Please set LastEditors
@Description: Convolutional Neural Networks for Sentence Classification
@FilePath: /textClassification/src/DL/models/CNN.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper
from __init__ import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed)
#         self.embedding = nn.Embedding(50000, 300)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
#         print(x.shape)
        out = self.embedding(x[0])
#         print(out.shape)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
