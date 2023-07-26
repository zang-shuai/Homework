import math
import numpy as np
import itertools
import copy
import time
import scipy.stats
from pure_ldp.frequency_oracles import DEClient
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import random

from BlackBox.DSubset.d_client import DSClient
from BlackBox.unary_encoding import UEClient
from BlackBox.utils import *

# from BlackAccess2.unary_encoding import UEClient
# from BlackAccess.utils import *
# from BlackAccess2.DSubset.d_client import DSClient

xxx = []


class countLDP_discrete(object):
    def __init__(self, mechanism, relevant=False):
        self.mechanism = mechanism
        self.relevant = relevant

    def count_probability(self, l1):
        l = list(l1)

        # l = []
        # for i in l1:
        #     l.append(list(i))
        #     pass
        # print(l)

        keys = set(l)

        d = dict()
        for k in keys:
            d[k] = l.count(k) / len(l)
        return d

    def JS_divergence(self, p, q):
        p = np.array(list(p.values()))
        q = np.array(list(q.values()))
        M = (p + q) / 2
        return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

    def similar_dictionary(self, dic1, dic2, err=0.01):
        if dic1.keys() != dic2.keys():
            return False
        else:
            JS = self.JS_divergence(dic1, dic2)
            # print(JS)

            if JS > 0.0001:
                return False
        return True

    def get_max(self, lmax, lmin):
        max_list = 0
        for i in range(len(lmax)):
            dict_max, dict_min = lmax[i], lmin[i]
            if self.similar_dictionary(dict_max, dict_min):
                continue
            max_data = -1
            for k in set(dict_max.keys()) & set(dict_min.keys()):
                # print(dict_max[k] , dict_min[k])
                # if k in dict_min.keys():
                if dict_max[k] == 0 or dict_min[k] == 0:
                    continue
                data = math.fabs(math.log(dict_max[k] / dict_min[k]))
                max_data = max(max_data, data)
            max_list += max_data
        return max_list

    def evaluation(self, num=10000):
        # 获取长度
        k = self.mechanism.d
        # 获取数据
        datas = range(1, k + 1)

        # 输出列表
        outputs = []
        datas = random.sample(datas, max(math.ceil(np.log(k)), 2))

        # 循环数据，分别将他们io，num次
        for data in datas:
            # a = self.mechanism.privatise(data)
            if self.relevant:
                # if isinstance(a, set):
                # output = []
                # for _ in range(num):
                #     output.extend(self.mechanism.privatise(data))
                # outputs.append(output)
                outputs.append([''.join(self.tostr(self.mechanism.privatise(data))) for _ in range(num)])
                # print('======')
                # print([data in self.mechanism.privatise(data) for _ in range(num)])
                # outputs.append([data in self.mechanism.privatise(data) for _ in range(num)])
            else:
                outputs.append([self.mechanism.privatise(data) for _ in range(num)])
        # 输出概率列表，里面存储k个字典，每个字典里有输入为x输出结果的集合与其相应概率
        outputs = np.array(outputs)
        output_pro = []
        for output in outputs:
            pro = []
            if isinstance(output[0], list) or isinstance(output[0], np.ndarray):
                for i in range(len(output[0])):
                    pro.append(self.count_probability(output[:, i]))
            else:
                pro.append(self.count_probability(output))

            output_pro.append(pro)

        # 获取所有的可能输出的结果
        il = list(itertools.combinations(range(len(datas)), 2))
        maxdatas = []
        for i in il:
            v1 = output_pro[i[0]]
            v2 = output_pro[i[1]]
            maxdatas.append(self.get_max(v1, v2))
        return max(maxdatas)

    def tostr(self, f):
        return [str(i) for i in f]


# @cost_time
def run1(epsilons, n=100, m=10000):
    for epsilon in epsilons:
        oue = UEClient(epsilon, 100, use_oue=True)
        coue = countLDP_discrete(oue)
        run_n(coue.evaluation, n=n, m=m, label='$\epsilon=$' + str(epsilon))
    plt.legend()
    plt.title('优化一元编码 (OUE) ')
    plt.show()


# @cost_time
def run2(epsilons, n=100, m=10000):
    for epsilon in epsilons:
        sue = UEClient(epsilon, 100, use_oue=False)
        csue = countLDP_discrete(sue)
        run_n(csue.evaluation, n=n, m=m, label='$\epsilon=$' + str(epsilon))
    plt.legend()
    plt.title('对称一元编码(SUE)')
    plt.show()


# @cost_time
def run3(epsilons=None, n=100, m=10000):
    for epsilon in epsilons:
        de = DEClient(epsilon, 5)
        cde = countLDP_discrete(de)
        run_n(cde.evaluation, n=n, m=m, label='$\epsilon=$' + str(epsilon))
    plt.legend()
    plt.title('随机响应')
    plt.show()


# @cost_time
def run4(epsilons, n=100, m=10000):
    for epsilon in epsilons:
        ds = DSClient(epsilon, d=6, k=3)
        cds = countLDP_discrete(ds, relevant=True)
        run_n(cds.evaluation, n=n, m=m, label='$\epsilon=$' + str(epsilon))
    plt.legend()
    plt.title('子集选择')
    plt.show()


if __name__ == '__main__':
    epsilons = [0.3, 0.5, 0.8]
    # oue
    run1(epsilons=epsilons, n=100, m=40000)
    # # sue
    run2(epsilons=epsilons, n=100, m=40000)
    # de
    # run3(epsilons=epsilons, n=100, m=10000)
    # # ds
    # run4(epsilons=epsilons, n=100, m=40000)
    # xxx = sorted(xxx)
    # print(xxx[979], xxx[980])
    # cdf(sorted(xxx))
    # print(len(xxx))
