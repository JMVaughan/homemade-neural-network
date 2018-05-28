import numpy as np

from neural_network.optimizers import Momentum, GradientDescent, RMSProp, Adam
from neural_network.parameter_initialization import small_random, xavier_relu, xavier_tanh
from neural_network.cost_functions import categorical_cross_entropy, binomial_cross_entropy


class _Layer:
    def __init__(self, output_n, dropout_rate, parameter_initialization, batch_norm=False):
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

        self.gamma = None
        self.beta = None

        self.z = None
        self.z_norm = None
        self.z_tilde = None
        self.a = None
        self.d = None

        self.mu = None
        self.var = None
        self.weighted_avg_dict = {}

        self.epsilon = 10 ** (-7)

        # Placeholder for optimizer object
        self.optimizer_param_dict = {}
        self.optimizer_string = ''
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

    def initialize_optimizer(self, optimizer_string):
        self.optimizer_string = optimizer_string

        optimizer_dict = {'GRADIENTDESCENT': GradientDescent,
                          'MOMENTUM': Momentum,
                          'RMSPROP': RMSProp,
                          'ADAM': Adam}

        optimizer = optimizer_dict[optimizer_string.upper()]

        self.optimizer_param_dict['w'] = optimizer(self.w.shape)
        self.optimizer_param_dict['b'] = optimizer(self.b.shape)

        if self.batch_norm:
            self.optimizer_param_dict['gamma'] = optimizer(self.gamma.shape)
            self.optimizer_param_dict['beta'] = optimizer(self.beta.shape)

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

        self.weighted_avg_dict['mu'] = 0
        self.weighted_avg_dict['var'] = 0

    def linear_forward(self, input_a):
        """ Compute linear part of activation: Z = WA + B """
        assert input_a.shape[0] == self.input_n

        z = np.dot(self.w, input_a) + self.b
        assert z.shape == (self.output_n, input_a.shape[1])
        return z

    def linear_activation_forward(self, input_a, train):
        """ Apply activation function """
        # Calculate linear part
        self.z = self.linear_forward(input_a)

        # Apply batch normalization
        if self.batch_norm:
            self.z_tilde = self.apply_batch_norm(self.z, train)
        else:
            self.z_tilde = self.z

        # Activate linear part
        self.a = self.activation(self.z_tilde)

        # Apply dropout
        self.a = self.apply_dropout(self.a)

        assert self.a.shape == (self.output_n, input_a.shape[1])
        return self.a

    def apply_batch_norm(self, z, train):
        self.z_norm = self.normalize_batch(z, train)
        z_tilde = self.z_norm*self.gamma + self.beta
        return z_tilde

    def normalize_batch(self, z, train):
        if train:
            self.mu = np.mean(z, axis=1, keepdims=True)
            self.var = np.var(z, axis=1, keepdims=True)
            self.keep_running_weighed_avg(self.mu, self.var)

        else:
            self.mu = self.weighted_avg_dict['mu']
            self.var = self.weighted_avg_dict['var']

        z_norm = (z - self.mu)/np.sqrt(self.var + self.epsilon)
        return z_norm

    def keep_running_weighed_avg(self, mu, var):
        beta = 0.9
        self.weighted_avg_dict['mu'] = beta*self.weighted_avg_dict['mu'] + (1 - beta)*mu
        self.weighted_avg_dict['var'] = beta*self.weighted_avg_dict['var'] + (1 - beta)*var

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
        self.w = self.optimizer_param_dict['w'].update_parameters(self.w, self.dw, learning_rate)
        self.b = self.optimizer_param_dict['b'].update_parameters(self.b, self.db, learning_rate)

        if self.batch_norm:
            self.gamma = self.optimizer_param_dict['gamma'].update_parameters(self.gamma, self.dgamma, learning_rate)
            self.beta = self.optimizer_param_dict['beta'].update_parameters(self.beta, self.dbeta, learning_rate)

    def set_parameter(self, array, param_type):
        """ Restore parameters """
        if param_type.upper() == "W":
            self.w = array
            self.input_n = self.w.shape[1]
        if param_type.upper() == "B":
            self.b = array

        if param_type.upper() == "GAMMA":
            self.gamma = array

        if param_type.upper() == "BETA":
            self.beta = array

        if param_type.upper() == "MU":
            self.weighted_avg_dict['mu'] = array

        if param_type.upper() == "VAR":
            self.weighted_avg_dict['var'] = array

        #ToDo make sure that mini bath generator handles last batch which is smaller than rest
        #ToDo refactor!
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
    def __init__(self, output_n, dropout_rate=0, parameter_initialization='XavierRelu'):
        super(SigmoidLayer, self).__init__(output_n, dropout_rate, parameter_initialization)
        self.cost_function = binomial_cross_entropy

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def activation_derivative(self, z):
        return self.a * (1 - self.a)

    def set_dz(self, a_l, y):
        da = - np.divide(y, a_l) + np.divide(1 - y, 1 - a_l)
        self.dz_tilde = np.multiply(da, self.activation_derivative(self.z))


class SoftMaxLayer(_Layer):
    def __init__(self, output_n, dropout_rate=0, parameter_initialization='XavierRelu'):
        super(SoftMaxLayer, self).__init__(output_n, dropout_rate, parameter_initialization)
        self.cost_function = categorical_cross_entropy

    def activation(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z/np.sum(e_z, axis=0)

    def activation_derivative(self, z):
        return self.a*(1 - self.a)

    def set_dz(self, a_l, y):
        # Categorical cross entropy
        self.dz_tilde = a_l - y
