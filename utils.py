__author__ = 'ayushgupta'
import random


def getdata(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f.readlines():
            if '?' not in line:
                row = line.strip('\n').split(',')
                data.append({
                    "attributes": row[1:-1],
                    "label": row[-1]
                })

    return data


def split(data, ratio):
    random.shuffle(data)

    return data[:int(ratio * len(data))], data[int(ratio * len(data)):]
