import pdb
import tensorflow as tf
from typing import List, Union
from models.tf2.layers import ExtendMask
from tensorflow.keras import Model
from tensorflow.keras import layers
from utils.IO import *
from models.tf2.mappings import activation_names
from models.tf2 import AbstractTf2Model


class LSTMNetwork(Model):
    """
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
        """
        """
        self.layer_size = layer_size
        self.dropout_rate = dropout
        self.recurrent_dropout = recurrent_dropout
        self.depth = depth

        if not final_activation in activation_names:
            raise ValueError(f"Activation function {final_activation} not supported. "
                             f"Must be one of {*activation_names,}")

        if isinstance(layer_size, int):
            self._hidden_sizes = [layer_size] * depth
        else:
            self._hidden_sizes = layer_size
            if depth != 1:
                warn_io("Specified hidden sizes and depth are not consistent. "
                        "Using hidden sizes and ignoring depth.")

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
            x = layers.Dropout(dropout)(x)

        if deep_supervision:
            y = layers.TimeDistributed(layers.Dense(output_dim, activation=final_activation))(x)
            y = ExtendMask()([y, M])
        else:
            y = layers.Dense(output_dim, activation=final_activation)(x)

        super(LSTMNetwork, self).__init__(inputs=inputs, outputs=y)


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
