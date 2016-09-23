__author__ = 'ayushgupta'

import matplotlib.pyplot as plt
import math


def plot(params, xlabel, ylabel):
    length = 1
    for param in params:
        x = []
        y = []
        if 'length' in param:
            length = max(param['length'], length)
        for key in sorted(param['accuracy'].keys()):
            x.append(key)
            y.append(param['accuracy'][key])
        if 'label' in param:
            plt.plot(x, y, marker='o', label=param['label'])
        else:
            plt.plot(x, y, marker='o')

    length = int(math.ceil(length / (10**(len(str(length))-1) * 1.0)) * (10**(len(str(length))-1)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([0, length, 0, 100])
    plt.legend(loc=0)

    return plt
