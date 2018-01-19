# coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt


class LoadIdx3():
    def __init__(self):
        pass

    def load_set_file(self, set_file="data/train-images.idx3-ubyte"):
        file = open(set_file, "rb")
        buf = file.read()
        index = 0
        magic, numImages, numRows, numColumns = struct.unpack_from(
            '>IIII', buf, index)
        index += struct.calcsize('>IIII')
        self.set = np.zeros((numImages, numRows * numColumns))
        j = 0
        while(index < len(buf)):
            self.set[j] = struct.unpack_from('>784B', buf, index)
            index += struct.calcsize('>784B')
            j += 1
        # im = self.set[59999]
        # im = im.reshape(28, 28)
        # fig = plt.figure()
        # plt.imshow(im, cmap='gray')
        # plt.show()
        # normallize 数据集
        set_max,set_min = self.set.max(),self.set.min()
        self.set = (self.set - set_min) / (set_max - set_min)
        return self.set

    def load_label_file(self, label_file="data/train-labels.idx1-ubyte"):
        file = open(label_file, "rb")
        buf = file.read()
        index = 0
        magic, numImages = struct.unpack_from(
            '>II', buf, index)
        index += struct.calcsize('>II')
        self.label = np.zeros((numImages, 1))
        j = 0
        while index < len(buf):
            self.label[j] = struct.unpack_from('>B', buf, index)
            index += struct.calcsize('>B')
            j += 1
        return self.label

# fd = LoadIdx3()
# fd.load_set_file("train-images.idx3-ubyte")
# fd.load_label_file("train-labels.idx1-ubyte")
# print(fd.label[59999])
