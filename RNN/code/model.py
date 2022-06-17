# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：model.py
@Author ：AnthonyZ
@Date ：2022/6/14 13:34
"""


import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # print(input_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # print(x.shape)
        # print(c_n.shape, h_n.shape, x.shape)
        out = (self.linear(out[-1]))
        # print(x[-1].shape)
        out = self.softmax(out)
        return out
