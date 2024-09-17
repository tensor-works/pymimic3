from models.tf2 import AbstractTf2Model
import tensorflow as tf
from collections import OrderedDict
from typing import List, Union
from models.tf2.layers import ExtendMask, GetTimestep, Slice
from utils.arrays import isiterable
from keras.api._v2.keras import layers
from utils.IO import *
from models.tf2.mappings import activation_names


class LSTMNetwork(AbstractTf2Model):
    """
    Long Short-Term Memory (LSTM) network model for time-series prediction tasks.

    This class implements an LSTM-based neural network with support for deep supervision,
    dropout, and multiple layers. It allows flexible configuration of the LSTM architecture
    by specifying layer sizes, input dimensions, and dropout rates.

    Parameters
    ----------
    layer_size : Union[List[int], int]
        The size of each LSTM layer. If a list is provided, each element corresponds to
        the number of units in a respective layer (bottom->top). If an integer is provided, all layers
        will have the same number of units.
    input_dim : int
        Dimensionality of the input data.
    dropout : float, optional
        The dropout rate applied to LSTM layers and the final output before the dense layer (default is 0.).
    deep_supervision : bool, optional
        If True, the model will apply deep supervision by returning outputs at each
        time step (default is False).
    recurrent_dropout : float, optional
        The dropout rate applied to the recurrent state within the LSTM layers (default is 0.).
    final_activation : str, optional
        The activation function to use in the final output layer. If not provided, "sigmoid"
        will be used for binary outputs (when `output_dim` is 1), and "softmax" will be used
        for multi-class outputs (default is 'linear').
    output_dim : int, optional
        Dimensionality of the model output (default is 1).
    depth : int, optional
        Number of LSTM layers (default is 1).

    """

    def __init__(self,
                 layer_size: Union[List[int], int],
                 input_dim: int,
                 dropout: float = 0.,
                 deep_supervision: bool = False,
                 recurrent_dropout: float = 0.,
                 final_activation: str = 'linear',
                 output_dim: int = 1,
                 depth: int = 1):
        # Parameters memory
        self._layer_size = layer_size
        self._dropout_rate = dropout
        self._recurrent_dropout = recurrent_dropout
        self._depth = depth
        self._deep_supervision = deep_supervision

        # Final activation resolution
        if final_activation is None:
            if output_dim == 1:
                self._final_activation = "sigmoid"
            else:
                self._final_activation = "softmax"
        else:
            if not final_activation in activation_names:
                raise ValueError(f"Activation function {final_activation} not supported. "
                                 f"Must be one of {*activation_names,}")

        # Input layers and masking
        X = layers.Input(shape=(None, input_dim), name='x')
        inputs = [X]
        x = layers.Masking()(X)

        if deep_supervision:
            M = layers.Input(shape=(None,), name='M')
            inputs.append(M)

        if isinstance(layer_size, int):
            iterator = [layer_size] * (depth - 1)
            last_layer_size = layer_size
        else:
            iterator = layer_size[:-1]
            last_layer_size = layer_size[-1]

        for i, size in enumerate(iterator):
            x = layers.LSTM(units=size,
                            activation='tanh',
                            return_sequences=True,
                            recurrent_dropout=recurrent_dropout,
                            dropout=dropout,
                            name=f"lstm_hidden_{i}")(x)

        x = layers.LSTM(units=last_layer_size,
                        activation='tanh',
                        dropout=dropout,
                        return_sequences=deep_supervision,
                        recurrent_dropout=recurrent_dropout)(x)

        # Output module of the network
        if dropout > 0:
            # TODO! took this over from legacy but doesn't seem to make any sense
            x = layers.Dropout(dropout)(x)

        if deep_supervision:
            y = layers.TimeDistributed(layers.Dense(output_dim, activation=final_activation))(x)
            y = ExtendMask()([y, M])
        else:
            y = layers.Dense(output_dim, activation=final_activation)(x)

        super(LSTMNetwork, self).__init__(inputs=inputs, outputs=y)


class CWLSTMNetwork(AbstractTf2Model):
    """
    """

    def __init__(self,
                 layer_size: Union[List[int], int],
                 clayer_size: Union[List[int], int],
                 input_dim: int,
                 channels: List[str],
                 dropout: float = 0.,
                 deep_supervision: bool = False,
                 recurrent_dropout: float = 0.,
                 final_activation: str = 'linear',
                 output_dim: int = 1,
                 depth: int = 1):
        # Parameters memory
        self._layer_size = layer_size
        self._clayer_size = clayer_size
        self._dropout = dropout
        self._recurrent_dropout = recurrent_dropout
        self._depth = depth
        self._deep_supervision = deep_supervision

        # Final activation resolution
        if final_activation is None:
            if output_dim == 1:
                self._final_activation = "sigmoid"
            else:
                self._final_activation = "softmax"
        else:
            if not final_activation in activation_names:
                raise ValueError(f"Activation function {final_activation} not supported. "
                                 f"Must be one of {*activation_names,}")

        # Channels without masking and one-hot encoding
        channels_names = set([channel.split("->")[0].strip(" ") \
                              for channel in channels \
                              if not "mask" in channel])

        # Channel name to index mapping
        indices = range(input_dim)
        self._channels = OrderedDict({
            channel: list(filter(lambda i: channels[i].find(channel) != -1, indices))
            for channel in channels_names
        })

        # Input layers and masking
        X = layers.Input(shape=(None, input_dim), name='X')
        inputs = [X]
        x = layers.Masking()(X)

        if deep_supervision:
            M = layers.Input(shape=(None,), name='M')
            inputs.append(M)

        # Block 1: Compute channel wise x
        # Iterator resolution
        if isiterable(clayer_size):
            iterator = clayer_size
        else:
            iterator = [clayer_size] * depth

        cx = OrderedDict()
        for channel_names, channel_indices in self._channels.items():
            ccx = Slice(channel_indices)(x)  # Current channel x
            for size in iterator:
                ccx = layers.LSTM(units=size,
                                  activation='tanh',
                                  return_sequences=True,
                                  dropout=recurrent_dropout)(ccx)
            cx[channel_names] = ccx

        # Block 2: Concatenated LSTM layers
        # Concatenate processed channels
        x = layers.Concatenate(axis=2)(list(cx.values()))

        # Iterator resolution
        if isinstance(layer_size, int):
            iterator = [layer_size] * (depth - 1)
            last_layer_size = layer_size
        else:
            iterator = layer_size[:-1]
            last_layer_size = layer_size[-1]

        for _, size in enumerate(iterator):

            x = layers.LSTM(units=size,
                            activation='tanh',
                            return_sequences=True,
                            dropout=recurrent_dropout)(x)

        x = layers.LSTM(units=last_layer_size,
                        activation='tanh',
                        return_sequences=deep_supervision,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout)(x)

        # Output module of the network
        if dropout:
            # TODO! took this over from legacy but doesn't seem to make any sense
            x = layers.Dropout(dropout)(x)

        if deep_supervision:
            y = layers.TimeDistributed(layers.Dense(output_dim, activation=final_activation))(x)
            y = ExtendMask()([y, M])
        else:
            y = layers.Dense(output_dim, activation=final_activation)(x)

        super(CWLSTMNetwork, self).__init__(inputs=inputs, outputs=y)


if __name__ == "__main__":
    import datasets
    from pathlib import Path
    from tests.tsettings import *
    from preprocessing.scalers import MinMaxScaler
    from generators.tf2 import TFGenerator
    from tensorflow.keras.optimizers import Adam

    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=Path(SEMITEMP_DIR),
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                deep_supervision=True,
                                impute_strategy='previous',
                                task="DECOMP")

    # reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)

    scaler = MinMaxScaler().fit_reader(reader)
    train_generator = TFGenerator(reader=reader,
                                  scaler=scaler,
                                  batch_size=8,
                                  shuffle=True,
                                  deep_supervision=True)

    # val_generator = TFGenerator(reader=reader.val, scaler=scaler, batch_size=8, shuffle=True)

    model_path = Path(TEMP_DIR, "tf_lstm")
    model_path.mkdir(parents=True, exist_ok=True)
    model = LSTMNetwork(1000,
                        59,
                        recurrent_dropout=0.,
                        output_dim=1,
                        depth=3,
                        final_activation='sigmoid')
    model.compile(optimizer=Adam(learning_rate=0.000001, clipvalue=1.0), loss="binary_crossentropy")
    history = model.fit(train_generator, epochs=1000)
