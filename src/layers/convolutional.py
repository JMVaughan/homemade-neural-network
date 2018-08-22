import numpy as np
from neural_network.layers.layer_base import _LayerBase
from neural_network.optimizers import GradientDescent, RMSProp, Momentum, Adam
from neural_network.parameter_initialization import small_random_convolution_filter
from scipy.signal import convolve2d

from src.activations import ReLuLayer, TanhLayer


class Convolutional(_LayerBase):
    param_layer = True
    def __init__(self, filter_number, filter_size, padding_type, stride, activation_type):
        self.output_dim = None
        self._input_dim = None

        self.first = False
        self.last = False

        self.a = None
        self.input_a = None
        self.z = None

        self.optimizer_string = ''

        self.filter_number = filter_number
        self.filter_channels = None
        self.filter_size = filter_size
        self.filter_size_x = filter_size[0]
        self.filter_size_y = filter_size[1]

        self.padding_size = None

        self.dropout_rate = 0

        self.parameters_loaded = False
        self.batch_norm = False

        self.parameter_initialization = ' '

        self.optimizer_param_dict = {}

        self.activation_type = activation_type
        self.activation_function = None
        self.initialize_activation()

        self.padding_type = padding_type.upper()
        self.stride = stride

        self.w = None
        self.b = None

        self.dw = None
        self.db = None

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim):
        # ToDo Calculate Output dims based on input, padding and filter number
        self._input_dim = input_dim
        self.filter_channels = input_dim[-1]
        self.padding_size = self.calculate_padding_size()
        self.output_dim = self.calculate_output_dim(input_dim)

    def calculate_output_dim(self, input_dim):
        return (int(1 + (input_dim[1] + 2*self.padding_size - self.filter_size_y)/self.stride),
                int(1 + (input_dim[0] + 2 * self.padding_size - self.filter_size_x) / self.stride),
                self.filter_number)

    def calculate_padding_size(self):
        pad = None
        if self.padding_type == 'SAME':
            pad = int((self.input_dim[0] * (self.stride - 1) - self.stride + self.filter_size_x)/2)

        # ((s - 1)size - s + f)/2 = p

        elif self.padding_type == 'VALID':
            pad = 0

        return pad

    def initialize_layer(self, optimizer):
        """ Initialize parameters and optimizer """

        # Initialize parameters
        if not self.parameters_loaded:
            self.initialize_parameters()

        self.initialize_optimizer(optimizer)

    def initialize_parameters(self):
        self.w = small_random_convolution_filter(self.filter_size_x,
                                                 self.filter_size_y,
                                                 self.filter_channels,
                                                 self.filter_number)
        self.b = np.zeros(shape=self.output_dim)

    def initialize_optimizer(self, optimizer_string):
        self.optimizer_string = optimizer_string

        optimizer_dict = {'GRADIENTDESCENT': GradientDescent,
                          'MOMENTUM': Momentum,
                          'RMSPROP': RMSProp,
                          'ADAM': Adam}

        optimizer = optimizer_dict[optimizer_string.upper()]

        self.optimizer_param_dict['w'] = optimizer(self.w.shape)
        self.optimizer_param_dict['b'] = optimizer(self.b.shape)

        # if self.batch_norm:
        #     self.optimizer_param_dict['gamma'] = optimizer(self.gamma.shape)
        #     self.optimizer_param_dict['beta'] = optimizer(self.beta.shape)

    def initialize_activation(self):
        activation_dict = {'RELU': ReLuLayer,
                           'TANH': TanhLayer}

        self.activation_function = activation_dict[self.activation_type.upper()]()

    def update_parameters(self, learning_rate):
        """ Apply update to parameters (gradient decent)"""
        self.w = self.optimizer_param_dict['w'].update_parameters(self.w, self.dw, learning_rate)
        self.b = self.optimizer_param_dict['b'].update_parameters(self.b, self.db, learning_rate)

    def forwards(self, input_a, train):
        """ Apply activation function """
        self.input_a = input_a

        # Calculate linear part
        self.z = self.linear_forward(input_a)

        # Activate linear part
        self.a = self.activation_function.activate(self.z)
        return self.z

    def linear_forward(self, input_a):
        z = np.zeros(shape=self.output_dim)
        for filter_no in np.arange(self.filter_number):
            conv_temp = np.zeros(shape=self.input_dim[:-1])
            for channel_no in np.arange(self.filter_channels):
                conv_temp += convolve2d(input_a[..., channel_no], self.w[..., channel_no, filter_no], mode=self.padding_type.lower())
            z[..., filter_no] = conv_temp

        return z

    def backwards(self, da_output=None, mini_batch_size=None, a_l=None, y=None):

        self.db = da_output
        self.dw = np.zeros_like(self.w)
        da_output_temp = np.zeros_like(self.input_a)

        self.dz = da_output * self.activation_function.activate_derivative(self.z)

        # dW
        for j in np.arange(self.dw.shape[-1]):
            for i in np.arange(self.dw.shape[-2]):
                self.dw[:, :, i, j] = convolve2d(np.pad(np.rot90(self.input_a[:, :, i], k=2, axes=(0, 1)), pad_width=self.padding_size, mode='constant', constant_values=(0, 0)), self.dz[:, :, j], mode='valid')

        if not self.last:
            # da_output back
            for i in np.arange(da_output_temp.shape[-1]):
                for j in np.arange(self.w.shape[-1]):

                    da_output_temp[:, :, i] += convolve2d(self.dz[1:-1, 1:-1, j],
                                                          np.rot90(self.w, k=2, axes=(0, 1))[:, :, i, j])

        return da_output_temp