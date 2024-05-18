import pdb
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional


class LSTMNetwork(Model):
    """
    """

    def __init__(self,
                 layer_size,
                 dropout_rate,
                 input_dim,
                 bidirectional=False,
                 recurrent_dropout=0.,
                 task=None,
                 target_repl=False,
                 output_dim=1,
                 depth=1):
        """
        """
        self.layer_size = layer_size
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.depth = depth

        final_activation = {
            "DECOMP": 'sigmoid',
            "IHM": 'sigmoid',
            "LOS": 'softmax',
            "PHENO": 'softmax',
            None: None if output_dim == 1 else 'softmax'
        }

        num_classes = {"DECOMP": 1, "IHM": 1, "LOS": 10, "PHENO": 25, None: output_dim}

        # Input layers and masking
        input = layers.Input(shape=(None, input_dim), name='x')

        x = layers.Masking()(input)

        # TODO: compare bidirectional runs to one directiona

        if type(layer_size) == int:
            iterator = [layer_size] * (depth - 1)
        else:
            iterator = layer_size[:-1]
            layer_size = layer_size[-1]

        for i, size in enumerate(iterator):
            if bidirectional:
                num_units = size // 2
                x = Bidirectional(
                    layers.LSTM(units=num_units,
                                activation='tanh',
                                return_sequences=True,
                                recurrent_dropout=recurrent_dropout,
                                dropout=dropout_rate,
                                name=f"lstm_hidden_{i}"))(x)
            else:
                x = layers.LSTM(units=size,
                                activation='tanh',
                                return_sequences=True,
                                recurrent_dropout=recurrent_dropout,
                                dropout=dropout_rate,
                                name=f"lstm_hidden_{i}")(x)

        # Output module of the network
        return_sequences = target_repl
        '''        
        x = Bidirectional(layers.LSTM(units=layer_size,
                                      activation='tanh',
                                      return_sequences=return_sequences,
                                      dropout=dropout_rate,
                                      recurrent_dropout=recurrent_dropout,
                                      name=f"lstm_hidden_{depth}"))(x)
        '''
        x = layers.LSTM(units=layer_size,
                        activation='tanh',
                        return_sequences=return_sequences,
                        dropout=dropout_rate,
                        recurrent_dropout=recurrent_dropout)(x)

        x = layers.Dense(num_classes[task], activation=final_activation[task])(x)

        super(LSTMNetwork, self).__init__(inputs=[input], outputs=[x])
