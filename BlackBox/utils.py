# utils:
# from pure_ldp.frequency_oracles import UEClient
import time
import matplotlib.pyplot as plt
import warnings
import numpy as np


def count_probability(l1):
    l = []
    if isinstance(l1[0], list) or isinstance(l1[0], np.ndarray):
        for strs in l1:
            string = ''
            for s in strs:
                string += str(s)
            l.append(string)
    else:
        l = l1
    keys = set(l)
    d = dict()
    for k in keys:
        d[k] = l.count(k) / len(l)
    return d


def cdf(ar, label):
    import seaborn as sns
    kwargs = {'cumulative': True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ar.append(1.0)
        sns.distplot(ar, hist_kws=kwargs, kde_kws=kwargs, hist=False, label=label)
    # plt.title()
    # plt.show()


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'函数 {func.__name__} 耗时:{time.perf_counter() - t:.8f} 秒')
        return result

    return fun


def run_n(fun,label, n=100, m=10000):
    ans = []
    for i in range(n):
        print(i, end=" ")
        data = fun(num=m)
        ans.append(float(data))
    print(ans)
    cdf(ans,label=label)
