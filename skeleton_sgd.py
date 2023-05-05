#################################
# Your name: Sean Zaretzky Id 209164086
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    Runs T stochastic gradient updates and returns the resulting w_T. 
    """
    # question 1a
    sampla_size = len(data)
    w = 0
    for t in range(1, T + 1):
        i = np.random.randint(0, sample_size)
        step = eta_0 / t
        xi = data[i]
        yi = labels[i]
        if yi * np.inner(w, xi) < 1:
            w = ((1-step)*w) + (step*C*yi*xi)
        else:
            w = (1-step)*w
    
    return w




def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    # TODO: Implement me
    pass

#################################

# Place for additional code

#################################


if __name__ == '__main__':
    # helper()
    # print(train_data[:10])
    # print(train_labels[:10])
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    # question 1a
    T = 1000
    C = 1