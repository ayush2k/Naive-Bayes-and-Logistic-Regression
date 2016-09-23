__author__ = 'ayushgupta'

import math

import utils


def calcpy(example, weights):
    submission = 0.0
    for i, attr in enumerate(example['attributes']):
        # Submission of w_i*x_i
        submission += weights[i + 1] * int(attr)

    return (math.exp(weights[0] + submission)) / (1 + math.exp(weights[0] + submission))


def train(trainingdata, eta, epsilon):
    if trainingdata is None or len(trainingdata) == 0:
        return
    features = len(trainingdata[0]['attributes'])
    weights = [0 for i in range(features + 1)]
    notoptimal = True

    # iterate till weights are converged
    while notoptimal:
        sum_weights = [0 for i in range(features + 1)]

        # iterates through each training example
        for i, example in enumerate(trainingdata):
            # calculates p(y=1|X, W) estimated
            py = calcpy(example, weights)

            # difference = y_i - py
            if example['label'] == "+1":
                difference = 1 - py
            if example['label'] == "-1":
                difference = 0 - py

            # submission of difference between y and P(y=1|X, W) estimated
            sum_weights[0] += difference
            for j, attr in enumerate(example['attributes']):
                sum_weights[j + 1] += int(attr) * difference

        check = False

        # update weights
        for i, w in enumerate(weights):
            if abs(eta * sum_weights[i]) > epsilon:
                check = True

            # new w = old_w + (eta * sum(x_i * (y-p(y=1|X, W))_estimated)
            weights[i] = w + eta * sum_weights[i]

        if check is False:
            notoptimal = False

    return weights


def test(testdata, weights):
    correct = 0
    total = 0.0

    # iterates through each test data
    for example in testdata:
        py = calcpy(example, weights)

        # increments correct prediction
        if py >= 0.5 and example['label'] == '+1':
            correct += 1
        elif py < 0.5 and example['label'] == '-1':
            correct += 1
        total += 1

    if total == 0.0:
        return 100
    return correct / total * 100


def traintest(data, param, iterations, eta, epsilon):
    results = []

    # repeatedly train and test for random training set for a given size
    for i in range(iterations):
        trainingset, testingset = utils.split(data, param)
        trained = train(trainingset, eta, epsilon)
        accuracy = test(testingset, trained)
        results.append(accuracy)

    return sum(results) / len(results)


def logisticregression(data, trainingparams, iterations, eta, epsilon):
    results = {}

    # iterates through each of the setting for training
    for param in trainingparams:
        results[int(param * len(data))] = traintest(data, param, iterations, eta, epsilon)

    return {
        'accuracy': results,
        'length': len(data),
    }
