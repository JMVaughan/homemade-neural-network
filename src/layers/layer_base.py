
class _LayerBase:

    def initialize_layer(self, optimizer):
        raise NotImplementedError

    def forwards(self, a_previous, train):
        raise NotImplementedError

    def backwards(self):
        raise NotImplementedError

    @property
    def input_dim(self):
        raise NotImplementedError

    @input_dim.setter
    def input_dim(self, input_dim):
        """ This function needs to calculate output dimensions (important for CNN)"""
        raise NotImplementedError
