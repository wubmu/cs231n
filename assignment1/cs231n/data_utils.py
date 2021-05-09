from __future__ import print_function

import numpy as np  # 数据处理
from builtins import range
from six.moves import cPickle as pickle  # 序列化和反序列化
import os  # 操作文件和目录
from imageio import imread  #
import platform  # 获得操作系统信息


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """
    加载单个batch的数据
    :param filename: 文件路径
    :return: X，Y
    """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")  # 把单个通道的数据放在1 2维度方便操作
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """
    加载cifar的所有数据
    :param ROOT: 文件目录
    :return: Xtr, Ytr, Xte, Yte
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    # 加载测试数据
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte
