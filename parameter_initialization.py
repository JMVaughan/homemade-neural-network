import numpy as np


def small_random(output_n, input_n):
    # Initialize weights
    w = np.random.randn(output_n, input_n) * 0.01
    # Initialize intersect
    b = np.zeros(shape=(output_n, 1))

    return w, b


def xavier_tanh(output_n, input_n):
    # Initialize weights
    w = np.random.randn(output_n, input_n) * np.sqrt(1/input_n)
    # Initialize intersect
    b = np.zeros(shape=(output_n, 1))

    return w, b


def xavier_relu(output_n, input_n):
    # Initialize weights
    w = np.random.randn(output_n, input_n) * np.sqrt(2/input_n)
    # Initialize intersect
    b = np.zeros(shape=(output_n, 1))

    return w, b