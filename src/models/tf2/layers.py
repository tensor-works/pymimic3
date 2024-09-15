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


class Slice(Layer):
    """ Slice 3D tensor by taking x[:, :, indices]
    """

    def __init__(self, indices, **kwargs):
        self.supports_masking = True
        self.indices = indices
        super(Slice, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if K.backend() == 'tensorflow':
            xt = tf.transpose(x, perm=(2, 0, 1))
            gt = tf.gather(xt, self.indices)
            return tf.transpose(gt, perm=(1, 2, 0))
        return x[:, :, self.indices]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.indices))

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def get_config(self):
        return {'indices': self.indices}


class GetTimestep(Layer):
    """ Takes 3D tensor and returns x[:, pos, :]
    """

    def __init__(self, pos=-1, **kwargs):
        self.pos = pos
        self.supports_masking = True
        super(GetTimestep, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, self.pos, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        return {'pos': self.pos}
