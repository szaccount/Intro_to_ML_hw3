#################################
# Your name: Sean Zaretzky Id 209164086
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

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
    sample_size = len(data)
    data_dimension = len(data[0])
    w = np.array([0] * data_dimension)
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

def linear_classifier(w_classifier, data_point):
    """
    Returns prediction for label to the data point based on the passed w.
    """
    if np.inner(w_classifier, data_point) >= 0:
        return 1
    else:
        return -1

def accuracy_check(w_classifier, validation_data, validation_labels):
    error_sum = 0
    for i in range(len(validation_data)):
        prediction = linear_classifier(w_classifier, validation_data[i])
        if prediction != validation_labels[i]:
            error_sum += 1
    empirical_error = error_sum / len(validation_data)
    return 1 - empirical_error

def question_1a(train_data, train_labels, T, C, num_runs, eta_options):
    """
    Returns best eta0, accuracies for the passed eta0 options on training of passed data
    """
    eta_accuracies = []
    best_eta0 = -1
    best_eta0_accuracy = 0
    for eta0 in eta_options:
        sum_accuracy = 0
        for run in range(num_runs):
            train_res = SGD_hinge(train_data, train_labels, C, eta0, T)
            sum_accuracy += accuracy_check(train_res, validation_data, validation_labels)
        avg_accuracy_for_eta = sum_accuracy / num_runs
        eta_accuracies.append(avg_accuracy_for_eta)
        if avg_accuracy_for_eta > best_eta0_accuracy:
            best_eta0 = eta0
            best_eta0_accuracy = avg_accuracy_for_eta
    
    print(f"{best_eta0=} with accuracy {best_eta0_accuracy}")

    return best_eta0, eta_accuracies

def question_1b(train_data, train_labels, T, eta0, num_runs, C_options):
    """
    Returns best C, accuracies for the passed C options on training of passed data
    """
    C_accuracies = []
    best_C = -1
    best_C_accuracy = 0
    for C_cand in C_options:
        sum_accuracy = 0
        for run in range(num_runs):
            train_res = SGD_hinge(train_data, train_labels, C_cand, eta0, T)
            sum_accuracy += accuracy_check(train_res, validation_data, validation_labels)
        avg_accuracy_for_C = sum_accuracy / num_runs
        C_accuracies.append(avg_accuracy_for_C)
        if avg_accuracy_for_C > best_C_accuracy:
            best_C = C_cand
            best_C_accuracy = avg_accuracy_for_C
    
    print(f"{best_C=} with accuracy {best_C_accuracy}")
    
    return best_C, C_accuracies


#################################


if __name__ == '__main__':
    # helper()
    # print(train_data[:10])
    # print(train_labels[:10])
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    
    # question 1a finding optimal eta0
    T = 1000
    C = 1
    num_runs = 10
    eta_options = np.logspace(-5, 5, 11)
    _, eta_accuracies = question_1a(train_data, train_labels, T, C, num_runs, eta_options)
    plt.figure(1)
    plt.title("Average accuracy on validation set as function of eta_0")
    plt.xlabel("eta_0")
    plt.xscale("log")
    plt.ylabel("Average accuracy")
    plt.plot(eta_options, eta_accuracies)

    eta_options = np.logspace(-1, 1, 11)
    best_eta0, eta_accuracies = question_1a(train_data, train_labels, T, C, num_runs, eta_options)
    plt.figure(2)
    plt.title("Average accuracy on validation set as function of eta_0 increased resolution 1")
    plt.xlabel("eta_0")
    plt.xscale("log")
    plt.ylabel("Average accuracy")
    plt.plot(eta_options, eta_accuracies)


    # question 1b finding optimal C
    C_options = np.logspace(-5, 5, 11)
    _, C_accuracies = question_1b(train_data, train_labels, T, best_eta0, num_runs, C_options)
    plt.figure(3)
    plt.title("Average accuracy on validation set as function of C")
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("Average accuracy")
    plt.plot(C_options, C_accuracies)

    C_options = np.logspace(-5, -3, 11)
    best_C, C_accuracies = question_1b(train_data, train_labels, T, best_eta0, num_runs, C_options)
    plt.figure(4)
    plt.title("Average accuracy on validation set as function of C increased resolution")
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("Average accuracy")
    plt.plot(C_options, C_accuracies)

    # question 1c
    T = 20000
    train_res = SGD_hinge(train_data, train_labels, best_C, best_eta0, T)
    plt.figure(5)
    plt.title("Resulting w as an image")
    plt.imshow(np.reshape(train_res, (28, 28)), interpolation="nearest")
    # training w on train and validate data
    # train_extended_data = train_data[:]
    # train_extended_data.extend(validation_data[:])
    # train_extended_labels = train_labels[:]
    # train_extended_labels.extend(validation_labels[:])
    # train_res_extended = SGD_hinge(train_extended_data, train_extended_labels, best_C, best_eta0, T)
    # plt.figure(6)
    # plt.title("Resulting w as an image on extended train data")
    # plt.imshow(np.reshape(train_res_extended, (28, 28)), interpolation="nearest")

    # # question 1d
    # accuracy = accuracy_check(train_res, test_data, test_labels)
    # print(f"Accuracy of the best classifier on the test set is: {accuracy}")

    plt.show()
