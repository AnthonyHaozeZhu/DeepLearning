# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：Net.py
@Author ：AnthonyZ
@Date ：2022/6/2 15:36
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.batch_normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normal(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sample=False):
        super(BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.down_sample:
            self.sample = DownSample(in_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample:
            x = self.sample(x)
        out = self.relu2(out + x)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.bn = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.basic1 = BasicBlock(32, 32, 1)
        self.basic2 = BasicBlock(32, 64, 2, True)
        self.basic3 = BasicBlock(64, 64, 1)
        self.basic4 = BasicBlock(64, 128, 2, True)
        self.basic5 = BasicBlock(128, 128, 1)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.basic1(x)
        x = self.basic2(x)
        x = self.basic3(x)
        x = self.basic4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu2(x)
        return self.fc2(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseBasic(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBasic, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=(1, 1)),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x):
        out = torch.cat((x, self.layer(x)), dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        block = []
        for i in range(6):
            block.append(DenseBasic(in_channels, growth_rate))
            in_channels += growth_rate
        self.denseblock = nn.Sequential(*block)

    def forward(self, x):
        return self.denseblock(x)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.denseblock1 = DenseBlock(32, 32)
        self.bn2 = nn.BatchNorm2d(224)
        self.conv1 = nn.Conv2d(224, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.avg1 = nn.AvgPool2d(2, stride=2)
        self.denseblock2 = DenseBlock(64, 64)
        # self.avg2 = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(7168, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.denseblock1(x)
        x = self.bn2(x)
        x = self.conv1(x)
        x = self.avg1(x)
        x = self.denseblock2(x)
        batch_size, channels, w, h = x.shape
        x = x.reshape(batch_size, channels * w * h)
        x = self.fc1(x)
        return x


