import math

import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import time
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import random

from itertools import chain

from BlackBox.utils import run_n

err = []


# 连续向量
class countLDP_continue(object):
    def __init__(self, mechanism, h=1, kernel='gaussian', N=100, min=0.02):
        self.mechanism = mechanism
        # 核密度估计带宽
        self.h = h
        # 核密度估计的核函数（Str）
        self.kernel = kernel

        self.N = N

        self.min = min

    # 将最小值托底
    def delmin(self, n):
        for ix in range(len(n)):
            if n[ix] < math.log(self.min):
                n[ix] = math.log(self.min)
        return n

    def getfx(self, data, C1, C2):
        x_model = self.kernel_density(data)
        sequence = np.linspace(-C1, C2, self.N)

        ell = self.delmin(x_model.score_samples(sequence[:, np.newaxis]))

        return ell

    def kernel_density(self, data):
        x_train = np.array(data)
        model = KernelDensity(bandwidth=self.h, kernel=self.kernel)
        model.fit(x_train[:, np.newaxis])
        return model

    def get_data(self, data, p):
        m = sorted(data)
        return m[int(len(data) * p)]

    def get_max(self, lmax, lmin):
        res = 0
        for i in range(len(lmax)):
            dict_max, dict_min = lmax[i], lmin[i]
            data = self.get_data(np.abs(dict_max - dict_min), 0.9)
            err.append(data)
            if data > 0.1:
                res += data
        return res

    def evaluation(self, num=600000):
        # 获取长度
        k = self.mechanism.d
        # 获取数据
        datas = range(1, k + 1)

        # 输出列表

        outputs = []

        # 循环数据，分别将他们io，num次
        for data in datas:
            outputs.append([self.mechanism.privatise(data) for _ in range(num)])
        # 输出概率列表，里面存储k个字典，每个字典里有输入为x输出结果的集合与其相应概率
        output_fx = []

        for output in outputs:
            pro = []
            output = np.array(output)

            out = sorted(list(chain.from_iterable(output)))

            C1 = out[int(len(output) * 0.05)]
            C2 = out[int(len(output) * 0.95)]

            for i in range(len(output[0])):
                pro.append(self.getfx(output[:, i], C1, C2))
            output_fx.append(pro)
        epsilon = 0

        for i, j in list(itertools.combinations(range(k), 2)):
            output_in, output_out = output_fx[i], output_fx[j]
            epsilon = self.get_max(output_in, output_out)
            break
        return epsilon


# @cost_time
def run(epsilons):
    from pure_ldp.frequency_oracles import HEClient
    for epsilon in epsilons:

        oue = HEClient(epsilon, 5)
        coue = countLDP_continue(oue)
        # run_n(coue.evaluation, n=100)
        run_n(coue.evaluation, n=100, label='$\epsilon=$' + str(epsilon))
    plt.legend()
    plt.title('直方图编码 (HE) ')
    plt.show()




if __name__ == '__main__':
    run((0.3,0.5,0.8))
