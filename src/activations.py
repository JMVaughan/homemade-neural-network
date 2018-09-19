import numpy as np

from neural_network.src.cost_functions import categorical_cross_entropy


class TanhLayer:
    def activate(self, z):
        return np.tanh(z)

    def activate_derivative(self, z):
        return 1 - np.square(np.tanh(z))


class ReLuLayer:
    def activate(self, z):
        return np.maximum(0, z)

    def activate_derivative(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z


class SigmoidLayer:
    def activate(self, z):
        return 1 / (1 + np.exp(-z))

    def activate_derivative(self, a):
        return a * (1 - a)

    def set_dz(self, a_l, y, z):
        da = - np.divide(y, a_l) + np.divide(1 - y, 1 - a_l)
        dz_tilde = np.multiply(da, self.activate_derivative(z))
        return dz_tilde


class SoftMaxLayer:
    def __init__(self):
        self.cost_function = categorical_cross_entropy

    def activate(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z/np.sum(e_z, axis=0)

    def activate_derivative(self, z, a):
        return a*(1 - a)

    def set_dz(self, a_l, y):
        # Categorical cross entropy
        return a_l - y