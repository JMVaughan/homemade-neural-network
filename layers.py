import numpy as np

from neural_network.optimizers import Momentum, GradientDescent, RMSProp, Adam
from neural_network.parameter_initialization import small_random, xavier_relu, xavier_tanh
from neural_network.cost_functions import  categorical_cross_entropy, binomial_cross_entropy


class Layer:
    def __init__(self, input_n, output_n, l, dropout_rate, parameter_initialization):
        self.input_n = input_n
        self.output_n = output_n

        self.parameters_loaded = False

        self.dropout_rate = dropout_rate
        self.keep_prob = 1 - dropout_rate

        self.id = l

        self.w = None
        self.b = None
        self.z = None
        self.a = None
        self.d = None
        # Gradients
        self.dw = None
        self.db = None
        # Type of parameter initialization
        self.parameter_initialization = parameter_initialization
        # Placeholder for optimizer object
        self.optimizer = None
        self.cost_function = None

        # Gradients
        self.dz = None
        self.da = None

    def initialize_layer(self, optimizer):
        """ Initialize weights to be small and random """

        # Initialize parameters
        if not self.parameters_loaded:
            self.initialize_parameters()

        # Initialize optimizer
        self.initialize_optimizer(optimizer)

        assert self.w.shape == (self.output_n, self.input_n)
        assert self.b.shape == (self.output_n, 1)

    def initialize_optimizer(self, optimizer):
        optimizer_dict = {'GRADIENTDESCENT': GradientDescent,
                          'MOMENTUM': Momentum,
                          'RMSPROP': RMSProp,
                          'ADAM': Adam}

        self.optimizer = optimizer_dict[optimizer.upper()](self.w.shape, self.b.shape)

    def initialize_parameters(self):
        initialization_dict = {'SMALLRANDOM': small_random,
                               'XAVIERTANH': xavier_tanh,
                               'XAVIERRELU': xavier_relu}

        parameter_initializer = initialization_dict[self.parameter_initialization.upper()]

        # Perform initialization of parameters
        self.w, self.b = parameter_initializer(self.output_n, self.input_n)

    def linear_forward(self, input_a):
        """ Compute linear part of activation: Z = WA + B """
        assert input_a.shape[0] == self.input_n

        z = np.dot(self.w, input_a) + self.b
        assert z.shape == (self.output_n, input_a.shape[1])
        return z

    def linear_activation_forward(self, input_a):
        """ Apply activation function """
        # Calculate linear part
        self.z = self.linear_forward(input_a)

        # Calculate dropout matrix
        self.d = self.create_dropout_matrix()

        # Activate linear part
        self.a = self.activation(self.z)

        # Apply dropout
        self.a = np.multiply(self.a, self.d)
        # Rescale to keep expected value same with or without dropout
        self.a = self.a / self.keep_prob

        assert self.a.shape == (self.output_n, input_a.shape[1])
        return self.a

    def create_dropout_matrix(self):
        """ Create dropout mask """
        dropout_matrix = np.random.rand(*self.z.shape)
        dropout_matrix = (dropout_matrix < self.keep_prob)

        return dropout_matrix

    def update_parameters(self, learning_rate):
        """ Apply update to parameters (gradient decent)"""
        self.w, self.b = self.optimizer.update_parameters(self.w, self.b, self.dw, self.db, learning_rate)

    def set_parameter(self, array, param_type):
        """ Restore parameters """
        if param_type.upper() == "W":
            self.w = array
            self.input_n = self.w.shape[1]
        if param_type.upper() == "B":
            self.b = array

    def activation_derivative(self, z):
        pass

    def activation(self, z):
        pass


class TanhLayer(Layer):
    def __init__(self, input_n, output_n, l, dropout_rate, parameter_initialization):
        if not parameter_initialization:
            parameter_initialization = 'XavierTanh'
        super(TanhLayer, self).__init__(input_n, output_n, l, dropout_rate, parameter_initialization)

    def activation(self, z):
        return np.tanh(z)

    def activation_derivative(self, z):
        return 1 - np.square(np.tanh(z))


class ReLuLayer(Layer):
    def __init__(self, input_n, output_n, l, dropout_rate, parameter_initialization):
        if not parameter_initialization:
            parameter_initialization = 'XavierRelu'
        super(ReLuLayer, self).__init__(input_n, output_n, l, dropout_rate, parameter_initialization)

    def activation(self, z):
        return np.maximum(0, z)

    def activation_derivative(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z


class SigmoidLayer(Layer):
    def __init__(self, input_n, output_n, l, dropout_rate, parameter_initialization):
        if not parameter_initialization:
            parameter_initialization = 'XavierRelu'
        super(SigmoidLayer, self).__init__(input_n, output_n, l, dropout_rate, parameter_initialization)

        self.cost_function = binomial_cross_entropy

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def activation_derivative(self, z):
        return self.a * (1 - self.a)

    def set_dz(self, a_l, y):
        da = - np.divide(y, a_l) + np.divide(1 - y, 1 - a_l)
        self.dz = np.multiply(da, self.activation_derivative(self.z))


class SoftMaxLayer(Layer):
    def __init__(self, input_n, output_n, l, dropout_rate, parameter_initialization):

        if not parameter_initialization:
            parameter_initialization = 'XavierRelu'

        super(SoftMaxLayer, self).__init__(input_n, output_n, l, dropout_rate, parameter_initialization)

        self.cost_function = categorical_cross_entropy

    def activation(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z/np.sum(e_z, axis=0)

    def activation_derivative(self, z):
        return self.a*(1 - self.a)

    def set_dz(self, a_l, y):
        self.dz = a_l - y
