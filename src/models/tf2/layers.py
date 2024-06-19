import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class ExtendMask(Layer):
    """ Inputs:      [X, M]
        Output:      X
        Output_mask: M
    """

    def __init__(self, add_epsilon=False, **kwargs):
        self.supports_masking = True
        self.add_epsilon = add_epsilon
        super(ExtendMask, self).__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        return x[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, input, *args, **kwargs):
        if self.add_epsilon:
            return input[1] + K.epsilon()
        return input[1]

    def get_config(self):
        return {'add_epsilon': self.add_epsilon}
