import numpy as np
from neural_network.optimizers import Momentum, GradientDescent, RMSProp, Adam
from neural_network.parameter_initialization import small_random, xavier_relu, xavier_tanh

from src.activations import ReLuLayer, TanhLayer, SoftMaxLayer, SigmoidLayer


class FullyConnected:
    def __init__(self, output_n, activation_type, dropout_rate=0, parameter_initialization='XavierRelu', batch_norm=False):
        self.batch_norm = batch_norm
        self.output_n = output_n

        self.activation_type = activation_type
        self.activation_function = None
        # Initialize activation
        self.initialize_activation()

        self.dropout_rate = dropout_rate
        self.keep_prob = 1 - dropout_rate

        # Type of parameter initialization
        self.parameter_initialization = parameter_initialization

        self.input_n = None
        self.parameters_loaded = False

        self.id = None
        self.last = False

        self.input_a = None

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
        if self.batch_norm:
            self.initialize_batch_norm_parameters()

        # Initialize optimizer
        self.initialize_optimizer(optimizer)

        assert self.w.shape == (self.output_n, self.input_n)
        assert self.b.shape == (self.output_n, 1)

    def initialize_activation(self):
        activation_dict = {'RELU': ReLuLayer,
                           'TANH': TanhLayer,
                           'SIGMOID': SigmoidLayer,
                           'SOFTMAX': SoftMaxLayer}

        self.activation_function = activation_dict[self.activation_type.upper()]()

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
        self.input_a = input_a

        # Calculate linear part
        self.z = self.linear_forward(input_a)

        # Apply batch normalization
        if self.batch_norm:
            self.z_tilde = self.apply_batch_norm(self.z, train)
        else:
            self.z_tilde = self.z

        # Activate linear part
        self.a = self.activation_function.activate(self.z_tilde)

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

    def back_prop(self, da_output=None, mini_batch_size=None, a_l=None, y=None):
        if self.last:
            # Start back-prop by getting dz_tilde from cost function
            self.dz_tilde = self.activation_function.set_dz(a_l, y)
        else:
            # da_dropout -> da
            da = self.drop_out_backwards(da_output)
            # da -> dz_tilde
            self.dz_tilde = self.calculate_dz_tilde(da)

        # dz_tilde -> dz
        self.dz = self.calculate_dz(mini_batch_size)
        # dw
        self.dw = self.calculate_dw()
        # db
        self.db = self.calculate_db()
        # d_input (i.e. output of previous layer)
        prev_da = self.calculate_previous_layer_da()

        return prev_da

    def calculate_dz(self, mini_batch_size):
        # Batch Norm
        if self.batch_norm:
            dz = self.batch_norm_backwards(mini_batch_size)
        else:
            dz = (1 / mini_batch_size)*self.dz_tilde
        return dz

    def batch_norm_backwards(self, mini_batch_size):
        """ Backwards propagation through batch normalization """
        self.dgamma = self.calculate_dgamma()
        self.dbeta = self.calculate_dbeta()
        dz = self.calculate_batch_norm_dz(mini_batch_size)
        return dz

    def calculate_batch_norm_dz(self, mini_batch_size):
        f_1 = (self.gamma/mini_batch_size)*(1/np.sqrt(self.var + self.epsilon))
        f_2 = (mini_batch_size*self.dz_tilde - self.dgamma*self.z_norm - self.dbeta)
        dz = f_1 * f_2
        return dz

    def calculate_dbeta(self):
        dbeta = np.sum(self.dz_tilde, axis=1, keepdims=True)
        return dbeta

    def calculate_dgamma(self):
        dgamma = np.sum(self.dz_tilde*self.z_norm, axis=1, keepdims=True)
        return dgamma

    def calculate_dw(self):
        dw = np.dot(self.dz, self.input_a.T)
        return dw

    def calculate_db(self):
        db = np.sum(self.dz, axis=1, keepdims=True)
        return db

    def calculate_dz_tilde(self, da_before_dropout):
        dz_tilde = da_before_dropout * self.activation_function.activate_derivative(self.z_tilde)
        return dz_tilde

    def drop_out_backwards(self, da):
        # Calculate derivative of cost, dZ, w.r.t. output Z of previous layer (l - 1)
        da_before_dropout = (da*self.d)/self.keep_prob

        return da_before_dropout

    def calculate_previous_layer_da(self):
        previous_layer_da = np.dot(self.w.T, self.dz)
        return previous_layer_da