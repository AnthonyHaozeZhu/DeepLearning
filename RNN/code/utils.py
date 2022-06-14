# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/6/14 14:38
"""

import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)