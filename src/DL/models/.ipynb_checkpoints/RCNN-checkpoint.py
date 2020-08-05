'''
@Author: https://github.com/649453932/
@LastEditTime: 2020-06-28 12:44:10
@LastEditors: Please set LastEditors
@Description: Recurrent Convolutional Neural Networks for Text Classification
@FilePath: /textClassification/src/DL/models/RCNN.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x[0])  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out) 
        out = out.squeeze()
        out = self.fc(out)
        return out