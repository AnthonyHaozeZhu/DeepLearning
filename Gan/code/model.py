# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：model.py
@Author ：AnthonyZ
@Date ：2022/6/14 15:00
"""
import torch
import torch.nn as nn


# class Discriminator(torch.nn.Module):
#     def __init__(self, inp_dim=784):
#         super(Discriminator, self).__init__()
#         self.linear1 = nn.Linear(inp_dim, 512)
#         self.linear2 = nn.Linear(512, 512)
#         self.linear3 = nn.Linear(512, 512)
#         self.linear4 = nn.Linear(512, 1)
#         self.drop_out = nn.Dropout(0.4)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.view(x.size(0), 784)  # flatten (bs x 1 x 28 x 28) -> (bs x 784)
#         x = self.linear1(x)
#         x = self.linear2(self.relu(x))
#         x = self.linear3(self.drop_out(self.relu(x)))
#         x = self.linear4(self.drop_out(self.relu(x)))
#         return torch.sigmoid(x)
#
#
# class Generator(nn.Module):
#     def __init__(self, z_dim=100):
#         super(Generator, self).__init__()
#         self.linear1 = nn.Linear(z_dim, 128)
#         self.linear2 = nn.Linear(128, 256)
#         self.batch_normal1 = nn.BatchNorm1d(256, 0.8)
#         self.linear3 = nn.Linear(256, 512)
#         self.batch_normal2 = nn.BatchNorm1d(512, 0.8)
#         self.linear4 = nn.Linear(512, 1024)
#         self.batch_normal3 = nn.BatchNorm1d(1024, 0.8)
#         self.linear5 = nn.Linear(1024, 784)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.linear2(self.relu(x))
#         x = self.linear3(self.relu(self.batch_normal1(x)))
#         x = self.linear4(self.relu(self.batch_normal2(x)))
#         out = self.linear5(self.relu(self.batch_normal3(x)))
#         out = self.tanh(out)
#         out = out.view(out.size(0), 1, 28, 28)
#         return out


class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), 784)  # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out) # range [-1, 1]
        # convert to image
        out = out.view(out.size(0), 1, 28, 28)
        return out
