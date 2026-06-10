import logging

import numpy as np


def test_default_rng():
    """测试默认随机数生成器是否正确。"""
    rng1 = np.random.default_rng(0)
    