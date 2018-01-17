import math
import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def logistic(a=1, x=None):
        rst = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            if x[i] > 100:
                rst[i] = 1
            elif x[i] < -100:
                rst[i] = 0
            else:
                rst[i] = 1 / (1 + math.exp(-a * x[i]))
        return rst

    @staticmethod
    def derivative_logistic(a=1, x=None):
        return a * x * (1 - x)

    @staticmethod
    def tanh(a=1.7159, b=2 / 3, x=None):
        rst = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            if x[i] > 100:
                rst[i] = 1
            elif x[i] < -100:
                rst[i] = -1
            else:
                rst[i] = a * math.tanh(b * x[i])
        return rst

    @staticmethod
    def derivative_tanh(a=1.7159, b=2 / 3, x=None):
        return (b / a) * (a - x) * (a + x)


    @staticmethod
    def relu(x=None):
        rst = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            rst[i] = x[i] if x[i] > 0 else 0
        return rst

    @staticmethod
    def derivative_relu(x=None):
        rst = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            rst[i] = 1 if x[i] > 0 else 0
        return rst
