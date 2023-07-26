# 子集选择（DSubset）
import copy
import random

import numpy as np


class DSClient(object):
    def __init__(self, epsilon, d, k):
        self.data = list(range(1, d + 1))
        self.epsilon = epsilon
        self.d = d
        self.k = k

    def encode(self, v):
        x = v
        return x

    def perhurb(self, x):
        p = (self.k * np.exp(self.epsilon)) / (self.k * np.exp(self.epsilon) + self.d - self.k)
        q = (self.k - p) / (self.d - 1)
        if isinstance(x, list):
            y = []
            for x_i in x:
                data2 = copy.copy(self.data)
                data2.remove(x_i)

                boo = random.choices([0, 1], weights=[p, 1 - p], k=1)[0]
                if boo == 0:
                    y_0 = random.sample(data2, self.k - 1)
                    y_0.append(x_i)
                else:
                    y_0 = random.sample(data2, self.k)
                y.append(y_0)
            return y
        else:
            data2 = copy.copy(self.data)
            data2.remove(x)

            boo = random.choices([0, 1], weights=[p, 1 - p], k=1)[0]
            if boo == 0:
                y = random.sample(data2, self.k - 1)
                y.append(x)
            else:
                y = random.sample(data2, self.k)
            # ans = np.zeros((self.d,),dtype='int')
            ans = [0 for _ in range(self.d)]
            for i in y:
                ans[i - 1] = 1
            # print(ans)
            return ans

    def privatise(self, v):
        x = self.encode(v)
        y = self.perhurb(x)
        return y

# ds = DSubset(list(range(6)), 0.3, 5)
#
# print(ds.run(3))
