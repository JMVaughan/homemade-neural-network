import numpy as np


def categorical_cross_entropy(m, a_l, y):
    return np.squeeze(- (1 / m) * np.sum(np.multiply(y, np.log(a_l))))


def binomial_cross_entropy(m, a_l, y):
    return np.squeeze(-(1/m)*np.sum(np.multiply(y, np.log(a_l)) + np.multiply(1 - y, np.log(1 - a_l))))

