from neural_network.layers.layer_base import _LayerBase


class Flatten(_LayerBase):
    param_layer = False

    def __init__(self):
        self._input_dim = None
        self.output_dim = None
        self.dropout_rate = 0
        self.optimizer_string = ''

    def initialize_layer(self, optimizer):
        pass

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim):
        self._input_dim = input_dim
        self.output_dim = self.calculate_output_dim()

    def calculate_output_dim(self):
        flatten_size = 1
        for i in self._input_dim:
            flatten_size = flatten_size*i

        return (flatten_size,)

    def forwards(self, a_previous, train):
        return a_previous.reshape(-1, 1)

    def backwards(self, da_output=None, mini_batch_size=None, a_l=None, y=None):
        return da_output.reshape(self.input_dim)

    def update_parameters(self, _):
        pass