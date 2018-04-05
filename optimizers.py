import numpy as np


class Optimizer:
    def __init__(self, w_shape, b_shape):
        raise NotImplementedError("No '__init__' method defined")

    def update_parameters(self, w, b, dw, db, learning_rate):
        raise NotImplementedError("No 'update_parameters' method defined")


class GradientDescent(Optimizer):
    def __init__(self, w_shape, b_shape):
        pass

    def update_parameters(self, w, b, dw, db, learning_rate):
        w -= learning_rate * dw
        b -= learning_rate * db
        return w, b


class Momentum(Optimizer):
    vdw = None
    vdb = None
    beta = None

    def __init__(self, w_shape, b_shape, beta=0.9):
        self.beta = beta
        self.vdw = np.zeros(w_shape)
        self.vdb = np.zeros(b_shape)

    def update_parameters(self, w, b, dw, db, learning_rate):
        self.vdw = self.beta * self.vdw + (1 - self.beta) * dw
        self.vdb = self.beta * self.vdb + (1 - self.beta) * db

        w -= learning_rate * self.vdw
        b -= learning_rate * self.vdb

        return w, b


class RMSProp(Optimizer):
    sdw = None
    sdb = None
    beta = None
    epsilon = 1 ** (-8)

    def __init__(self, w_shape, b_shape, beta=0.9):
        self.beta = beta
        self.sdw = np.zeros(w_shape)
        self.sdb = np.zeros(b_shape)

    def update_parameters(self, w, b, dw, db, learning_rate):

        self.sdw = self.beta * self.sdw + (1 - self.beta) * np.square(dw)
        self.sdb = self.beta * self.sdb + (1 - self.beta) * np.square(db)

        w -= learning_rate * dw / (np.sqrt(self.sdw) + self.epsilon)
        b -= learning_rate * db / (np.sqrt(self.sdb) + self.epsilon)

        return w, b


class Adam(Optimizer):
    sdw = None
    sdb = None

    vdw = None
    vdb = None

    beta1 = None
    beta2 = None

    epsilon = 1 ** (-8)

    def __init__(self, w_shape, b_shape, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2

        self.sdw = np.zeros(w_shape)
        self.sdb = np.zeros(b_shape)

        self.vdw = np.zeros(w_shape)
        self.vdb = np.zeros(b_shape)

        self.t = 1

    def update_parameters(self, w, b, dw, db, learning_rate):

        self.vdw = self.beta1 * self.vdw + (1 - self.beta1) * dw
        self.vdb = self.beta1 * self.vdb + (1 - self.beta1) * db

        self.sdw = self.beta2 * self.sdw + (1 - self.beta2) * np.square(dw)
        self.sdb = self.beta2 * self.sdb + (1 - self.beta2) * np.square(db)

        vdw_corrected = self.vdw / (1 - self.beta1 ** self.t)
        vdb_corrected = self.vdb / (1 - self.beta1 ** self.t)

        sdw_corrected = self.sdw / (1 - self.beta2 ** self.t)
        sdb_corrected = self.sdb / (1 - self.beta2 ** self.t)

        w -= learning_rate * vdw_corrected / (np.sqrt(sdw_corrected) + self.epsilon)
        b -= learning_rate * vdb_corrected / (np.sqrt(sdb_corrected) + self.epsilon)

        return w, b
