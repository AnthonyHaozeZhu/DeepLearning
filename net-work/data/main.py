# -*- coding: UTF-8 -*-
"""
@Project ：net-work 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/3/18 20:39
"""

import torch
import torch.nn as nn

import numpy as np
import  matplotlib.pyplot as plt


class Net(nn.Module):
    """
    定义网络结构
    """
    def __init__(self):
        """
        初始化网络结构
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)



if __name__ == "__main__":
    print("init!")
