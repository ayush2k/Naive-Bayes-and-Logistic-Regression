__author__ = 'ayushgupta'

import naivebayes
import logisticregression
import graph
import utils


def main():
    trainingparams = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
    iterations = 5
    eta = 0.001
    epsilon = 0.001
    valuerange = 10
    params = []

    data = utils.getdata('breast-cancer-wisconsin.data.txt')
    nb = naivebayes.naivebayes(data, trainingparams, iterations, valuerange)
    nb['label'] = 'Naive Bayes'
    lr = logisticregression.logisticregression(data, trainingparams, iterations, eta, epsilon)
    lr['label'] = 'Logistic Regression'
    params.append(lr)
    params.append(nb)
    plot = graph.plot(params, 'Training Set', 'Accuracy')
    plot.show()

    return


if __name__ == '__main__':
    main()
