# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/6/2 16:28
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import importlib
import logging


def create_dataloader(opt):
    dataloader = importlib.import_module("dataset." + opt.dataname)
    return dataloader.ImageTextDataloader(opt)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def init_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    # 使用FileHandler输出到文件
    num_list = []
    for filename in os.listdir('./log'):
        if 'log' not in filename:
            continue
        num = int(filename.split('.')[0][3:])
        num_list.append(num)
    num = max(num_list) + 1 if num_list != [] else 1
    fh = logging.FileHandler('./log/log{}.txt'.format(num))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(fh)
    return logger


def args_logger(opt: dict, logdir):
    with open(logdir, "w") as f:
        for k, v in opt.items():
            f.write(k + " : " + v + "\n")


