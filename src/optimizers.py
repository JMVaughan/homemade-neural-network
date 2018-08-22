import numpy as np


class Optimizer:
    def __init__(self, param_shape):
        raise NotImplementedError("No '__init__' method defined")

    def update_parameters(self, param, dparam, learning_rate):
        raise NotImplementedError("No 'update_parameters' method defined")


class GradientDescent(Optimizer):
    def __init__(self, param_shape):
        pass

    def update_parameters(self, param, dparam, learning_rate):
        param -= learning_rate * dparam
        return param


class Momentum(Optimizer):
    vdparam = None
    beta = None

    def __init__(self, param_shape, beta=0.9):
        self.beta = beta
        self.vdparam = np.zeros(param_shape)

    def update_parameters(self, param, dparam, learning_rate):
        self.vdparam = self.beta * self.vdparam + (1 - self.beta) * dparam

        param -= learning_rate * self.vdparam

        return param


class RMSProp(Optimizer):
    sdparam = None
    beta = None
    epsilon = 1 ** (-8)

    def __init__(self, param_shape, beta=0.9):
        self.beta = beta
        self.sdparam = np.zeros(param_shape)

    def update_parameters(self, param, dparam, learning_rate):

        self.sdparam = self.beta * self.sdparam + (1 - self.beta) * np.square(dparam)

        param -= learning_rate * dparam / (np.sqrt(self.sdparam) + self.epsilon)

        return param


class Adam(Optimizer):
    sdparam = None
    vdparam = None

    beta1 = None
    beta2 = None

    epsilon = 1 ** (-8)

    def __init__(self, param_shape, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2

        self.sdparam = np.zeros(param_shape)

        self.vdparam = np.zeros(param_shape)

        self.t = 1

    def update_parameters(self, param, dparam, learning_rate):

        self.vdparam = self.beta1 * self.vdparam + (1 - self.beta1) * dparam
        self.sdparam = self.beta2 * self.sdparam + (1 - self.beta2) * np.square(dparam)

        vdparam_corrected = self.vdparam / (1 - self.beta1 ** self.t)
        sdparam_corrected = self.sdparam / (1 - self.beta2 ** self.t)

        param -= learning_rate * vdparam_corrected / (np.sqrt(sdparam_corrected) + self.epsilon)

        return param
