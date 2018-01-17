# coding:utf-8
from ReadImage import LoadIdx3
from Bp import Bp
import numpy as np
import datetime

class Test:
    def __init__(self):
        pass

    def test(self, set="t10k-images.idx3-ubyte",
             set_label="t10k-labels.idx1-ubyte"):
        load_data = LoadIdx3()
        bp = Bp(set=load_data.load_set_file(),
                set_label=load_data.load_label_file())
        set = load_data.load_set_file(set_file=set)
        set_label = load_data.load_label_file(label_file=set_label)
        __set = np.insert(set, 0, values=1, axis=1)
        __set_label = set_label
        w0, w1, w2 = bp.iteration()
        count = 0
        for i in range(10000):
            x_in = __set[i].reshape(785, 1)
            v0, y0 = bp.calculate(w0, x=x_in)
            y0_add_offset = np.insert(y0, 0, values=1, axis=0)
            v1, y1 = bp.calculate(w1, x=y0_add_offset)
            y1_add_offset = np.insert(y1, 0, values=1, axis=0)
            v2, y2 = bp.calculate(w2, x=y1_add_offset)
            index = self.__max(y2)
            count = count + 1 if index == __set_label[i] else count
        return count

    def __max(self, y=None):
        rst = 0
        for i in range(10):
            rst = rst if y[i] < y[rst] else i
        return rst

start = datetime.datetime.now()
t = Test()
print(t.test())
end = datetime.datetime.now()
print(end-start)
# print(t.test(set="train-images.idx3-ubyte",set_label="train-labels.idx1-ubyte"))
