import numpy as np

from neural_network.optimizers import Momentum, GradientDescent, RMSProp, Adam
from neural_network.parameter_initialization import small_random, xavier_relu, xavier_tanh
from neural_network.cost_functions import categorical_cross_entropy, binomial_cross_entropy


class _Layer:
    def __init__(self, output_n, dropout_rate, parameter_initialization, batch_norm):
        self.batch_norm = batch_norm
        self.output_n = output_n

        self.dropout_rate = dropout_rate
        self.keep_prob = 1 - dropout_rate

        # Type of parameter initialization
        self.parameter_initialization = parameter_initialization

        self.input_n = None
        self.parameters_loaded = False

        self.id = None

        self.w = None
        self.b = None
        self.z = None
        self.z_norm = None
        self.z_tilde = None
        self.a = None
        self.d = None

        self.mu = None
        self.sig_squared = None

        self.gamma = None
        self.beta = None

        self.epsilon = 10 ** (-7)

        # Placeholder for optimizer object
        self.optimizer = None
        self.cost_function = None

        # Gradients
        self.dz = None
        self.dz_tilde = None
        self.da = None
        self.dw = None
        self.db = None
        self.dgamma = None
        self.dbeta = None

    def initialize_layer(self, optimizer):
        """ Initialize parameters and optimizer """

        # Initialize parameters
        if not self.parameters_loaded:
            self.initialize_parameters()

        # # Initialize batch-normalization parameters
        self.initialize_batch_norm_parameters()

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

    def initialize_batch_norm_parameters(self):
        self.gamma = np.ones(shape=(self.output_n, 1))
        self.beta = np.zeros(shape=(self.output_n, 1))

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

        # Apply batch normalization
        if self.batch_norm:
            self.z_tilde = self.apply_batch_norm(self.z)
        else:
            self.z_tilde = self.z

        # Activate linear part
        self.a = self.activation(self.z_tilde)

        # Apply dropout
        self.a = self.apply_dropout(self.a)

        assert self.a.shape == (self.output_n, input_a.shape[1])
        return self.a

    def apply_batch_norm(self, z):
        self.z_norm = self.normalize_batch(z)
        z_tilde = self.z_norm*self.gamma + self.beta
        return z_tilde

    def normalize_batch(self, z):
        self.mu = np.mean(z, axis=1, keepdims=True)
        self.sig_squared = np.var(z, axis=1, keepdims=True)
        z_norm = (z - self.mu)/np.sqrt(self.sig_squared+self.epsilon)
        return z_norm

    def apply_dropout(self, a):
        # Calculate dropout matrix
        self.d = self.create_dropout_matrix()
        temp_a = np.multiply(a, self.d)
        # Rescale to keep expected value same with or without dropout
        return temp_a/self.keep_prob

    def create_dropout_matrix(self):
        """ Create dropout mask """
        dropout_matrix = np.random.rand(*self.z.shape)
        dropout_matrix = (dropout_matrix < self.keep_prob)

        return dropout_matrix

    def update_parameters(self, learning_rate):
        """ Apply update to parameters (gradient decent)"""
        self.w, self.b = self.optimizer.update_parameters(self.w, self.b,
                                                          self.dw, self.db, learning_rate)
        #ToDo make this adam or something
        #ToDo workout how to do a forward pass (weighted averages)
        #ToDo make sure that mini bath generator handles last batch which is smaller than rest
        #ToDo refactor!

        if self.batch_norm:
            self.gamma -= learning_rate*self.dgamma
            self.beta -= learning_rate*self.dbeta

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


class TanhLayer(_Layer):
    def __init__(self, output_n, dropout_rate=0, parameter_initialization='XavierTanh', batch_norm=True):
        super(TanhLayer, self).__init__(output_n, dropout_rate, parameter_initialization, batch_norm)

    def activation(self, z):
        return np.tanh(z)

    def activation_derivative(self, z):
        return 1 - np.square(np.tanh(z))


class ReLuLayer(_Layer):
    def __init__(self, output_n, dropout_rate=0, parameter_initialization='XavierRelu', batch_norm=True):
        super(ReLuLayer, self).__init__(output_n, dropout_rate, parameter_initialization, batch_norm)

    def activation(self, z):
        return np.maximum(0, z)

    def activation_derivative(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z


class SigmoidLayer(_Layer):
    def __init__(self, output_n, dropout_rate=0, parameter_initialization='XavierRelu', batch_norm=False):
        super(SigmoidLayer, self).__init__(output_n, dropout_rate, parameter_initialization, batch_norm)
        self.cost_function = binomial_cross_entropy

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def activation_derivative(self, z):
        return self.a * (1 - self.a)

    def set_dz(self, a_l, y):
        da = - np.divide(y, a_l) + np.divide(1 - y, 1 - a_l)
        self.dz_tilde = np.multiply(da, self.activation_derivative(self.z))


class SoftMaxLayer(_Layer):
    def __init__(self, output_n, dropout_rate=0, parameter_initialization='XavierRelu', batch_norm=False):
        super(SoftMaxLayer, self).__init__(output_n, dropout_rate, parameter_initialization, batch_norm)
        self.cost_function = categorical_cross_entropy

    def activation(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z/np.sum(e_z, axis=0)

    def activation_derivative(self, z):
        return self.a*(1 - self.a)

    def set_dz(self, a_l, y):
        # Categorical cross entropy
        self.dz_tilde = a_l - y
