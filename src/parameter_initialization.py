import numpy as np


def small_random(output_n, input_n):
    # Initialize weights
    w = np.random.randn(output_n, input_n) * 0.01
    # Initialize intersect
    b = np.zeros(shape=(output_n, 1))

    return w, b


def small_random_convolution_filter(filter_size_x, filter_size_y, channel_n, filter_n):
    # Initialize weights
    w = np.random.randn(filter_size_x, filter_size_y, channel_n, filter_n) * 0.01

    return w


def small_random_convolution_intersect(x, y, filter_n):
    # Initialize weights
    b = np.zeros(shape=(y, x, filter_n)) * 0.01

    return b

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