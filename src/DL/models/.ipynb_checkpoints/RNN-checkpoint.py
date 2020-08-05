'''
@Author: https://github.com/649453932/
@Date: 2020-04-09 17:34:05
@LastEditTime: 2020-06-30 14:13:17
@LastEditors: Please set LastEditors
@Description: Recurrent Neural Network for Text Classification with Multi-Task Learning
@FilePath: /textClassification/src/DL/models/RNN.py
'''
import torch
import torch.nn as nn
import numpy as np
from __init__ import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
