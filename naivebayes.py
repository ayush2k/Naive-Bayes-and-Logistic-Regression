__author__ = 'ayushgupta'

import utils


def train(trainingdata, valuerange):
    if trainingdata is None or len(trainingdata) == 0:
        return
    negative = 0
    positive = 0

    # initializes dictionary to store probability for each attribute and label
    # py = P(Y), px = P(X|Y)
    pTable = {
        'positive': {
            'py': 0,
            'px': [],
        },
        'negative': {
            'py': 0,
            'px': [],
        },
    }

    # initializes bin for X
    dict = {}
    for i in range(1, valuerange + 1):
        dict[str(i)] = 1
    dict['total'] = valuerange

    for attr in trainingdata[0]['attributes']:
        pTable['positive']['px'].append(dict.copy())
        pTable['negative']['px'].append(dict.copy())

    # iterates through each training example
    for example in trainingdata:
        # increments bin for each attribute of example
        if example['label'] == '+1':
            positive += 1
            for i, attr in enumerate(example['attributes']):
                pTable['positive']['px'][i][attr] += 1
                pTable['positive']['px'][i]['total'] += 1
        if example['label'] == '-1':
            negative += 1
            for i, attr in enumerate(example['attributes']):
                pTable['negative']['px'][i][attr] += 1
                pTable['negative']['px'][i]['total'] += 1

    # calculate p(y)
    pTable['positive']['py'] = (positive + 10) / ((len(trainingdata) + valuerange * 2) * 1.0)
    pTable['negative']['py'] = (negative + 10) / ((len(trainingdata) + valuerange * 2) * 1.0)

    # calculate p(x|y= 1)
    for attr in pTable['positive']['px']:
        for val in attr:
            attr[val] = attr[val] / (attr['total'] * 1.0)

    # calculate p(x|y= 0)
    for attr in pTable['negative']['px']:
        for val in attr:
            attr[val] = attr[val] / (attr['total'] * 1.0)

    return pTable


def test(testdata, trained):
    total = 0.0
    correct = 0

    # iterates through each test data
    for example in testdata:
        total += 1

        # calculates P(Y=0)P(X|Y=0)
        ppositive = trained['positive']['py']
        for i, attr in enumerate(example['attributes']):
            ppositive *= trained['positive']['px'][i][attr]

        # calculates P(Y=0)P(X|Y=0)
        pnegative = trained['negative']['py']
        for i, attr in enumerate(example['attributes']):
            pnegative *= trained['negative']['px'][i][attr]

        # increments correct prediction
        if ppositive >= pnegative and example['label'] == '+1':
            correct += 1
        elif ppositive < pnegative and example['label'] == '-1':
            correct += 1

    if total == 0.0:
        return 100

    return correct / total * 100


def traintest(data, param, iterations, valuerange):
    results = []

    # repeatedly train and test for random training set for a given size
    for i in range(iterations):
        trainingset, testingset = utils.split(data, param)
        trained = train(trainingset, valuerange)
        accuracy = test(testingset, trained)
        results.append(accuracy)

    return sum(results) / len(results)


def naivebayes(data, trainingparams, iterations, valuerange):
    results = {}

    # iterates through each of the setting for training
    for param in trainingparams:
        results[int(param * len(data))] = traintest(data, param, iterations, valuerange)

    return {
        'accuracy': results,
        'length': len(data),
    }

