# coding:utf-8
import math
import numpy as np
from sigmoid import Sigmoid

W0_NUM = 50  # 第一层神经元个数
W1_NUM = 30  # 第二层神经元个数
OUT_NUM = 10  # 输出层神经元个数
IMG_COUNT = 60000  # 训练样本集的数量
EPOCH = 100  # 训练样本集的次数
BATCH_SIZE = 200  # 一次训练的样本个数
ITERATION = int(IMG_COUNT / BATCH_SIZE)  # 一个样本集迭代的次数


class Bp():
    def __init__(self, eta=0.0001, set=None, set_label=None):
        self.__eta = eta
        self.__set = np.insert(set, 0, values=1, axis=1)
        self.__set_label = set_label
        col = self.__set.shape[1]
        self.__w0 = np.random.normal(loc=0, scale=1 / math.sqrt(col), size=(W0_NUM, col))
        self.__w1 = np.random.normal(loc=0, scale=1 / math.sqrt(W0_NUM), size=(W1_NUM, W0_NUM + 1))
        self.__w2 = np.random.normal(loc=0, scale=1 / math.sqrt(W1_NUM), size=(OUT_NUM, W1_NUM + 1))

    def iteration(self):
        for x in range(EPOCH):
            for i in range(ITERATION):
                var_w2 = 0
                var_w1 = 0
                var_w0 = 0
                for b in range(BATCH_SIZE):
                    x_in = self.__set[i * BATCH_SIZE + b].reshape(785, 1)
                    v0, y0 = self.calculate(w=self.__w0, x=x_in)
                    y0_add_offset = np.insert(y0, 0, values=1, axis=0)
                    v1, y1 = self.calculate(w=self.__w1, x=y0_add_offset)
                    y1_add_offset = np.insert(y1, 0, values=1, axis=0)
                    v2, y2 = self.calculate(w=self.__w2, x=y1_add_offset)
                    d = np.zeros((10, 1)) - 1
                    d_index = int(self.__set_label[i * BATCH_SIZE + b])
                    d[d_index] = 1
                    delta_w2 = self.calculate_delta(d, y2)
                    var_w2 = var_w2 + self.update_output(
                        self.__w2, delta_w2, y1_add_offset)
                    delta_w1 = self.calculate_sum_delta(delta_w2, self.__w2)
                    var_w1 = var_w1 + self.update_hiding(
                        self.__w1, delta_w1, y0_add_offset, y1)
                    delta_w0 = self.calculate_sum_delta(
                        np.delete(delta_w1, 0, axis=0), self.__w1)
                    var_w0 = var_w0 + self.update_hiding(
                        self.__w0, delta_w0, x_in, y0)
                self.__w2 = self.__w2 + (1 / BATCH_SIZE) * self.__eta * var_w2
                self.__w1 = self.__w1 + (1 / BATCH_SIZE) * self.__eta * var_w1
                self.__w0 = self.__w0 + (1 / BATCH_SIZE) * self.__eta * var_w0
        return self.__w0, self.__w1, self.__w2

    def calculate(self, w=None, x=None):
        v = w.dot(x)
        y = Sigmoid.tanh(a=1, b=1, x=v)
        return v, y

    def calculate_delta(self, d, y):
        return (d - y) * Sigmoid.derivative_tanh(a=1, b=1, x=y)

    def calculate_sum_delta(self, delta, w):
        return (w.T).dot(delta)

    def update_output(self, w, delta, x):
        # print((self.__eta*delta.dot(x.T)).shape)
        return delta.dot(x.T)

    def update_hiding(self, w, sum_delta, x, y):
        return (Sigmoid.derivative_tanh(a=1, b=1, x=y)
                * np.delete(sum_delta, 0, axis=0)).dot(x.T)
