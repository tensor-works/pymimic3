import pdb
import tensorflow as tf
from tensorflow.keras.layers import ExtendMask
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional


class LSTMNetwork(Model):
    """
    """

    def __init__(self,
                 layer_size,
                 dropout,
                 input_dim,
                 recurrent_dropout=0.,
                 deep_supervision=False,
                 task=None,
                 output_dim=1,
                 depth=1):
        """
        """
        self.layer_size = layer_size
        self.dropout_rate = dropout
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

        # Output module of the network
        x = layers.LSTM(units=last_layer_size,
                        activation='tanh',
                        return_sequences=False,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout)(x)

        if dropout > 0:
            x = layers.Dropout(dropout)(x)

        if deep_supervision:
            y = layers.TimeDistributed(
                layers.Dense(num_classes[task], activation=final_activation[task]))(x)
            y = ExtendMask()([y, M])
        else:
            y = layers.Dense(num_classes[task], activation=final_activation[task])(x)

        super(LSTMNetwork, self).__init__(inputs=[X], outputs=[x])


if __name__ == "__main__":
    import datasets
    from pathlib import Path
    from tests.tsettings import *
    from preprocessing.scalers import MinMaxScaler
    from generators.tf2 import TFGenerator
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task="IHM")

    reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)

    scaler = MinMaxScaler().fit_reader(reader.train)
    train_generator = TFGenerator(reader=reader.train, scaler=scaler, batch_size=2, shuffle=True)
    val_generator = TFGenerator(reader=reader.val, scaler=scaler, batch_size=2, shuffle=True)

    import torch
    import torch.nn as nn
    import torch.optim as optim

    model_path = Path(TEMP_DIR, "tf_lstm")
    model_path.mkdir(parents=True, exist_ok=True)
    model = LSTMNetwork(10, 0.2, 59, recurrent_dropout=0., output_dim=1, depth=2)

    model.compile(optimizer="adam", loss="mse")
    # Example training loop
    history = model.fit(train_generator, validation_data=val_generator, epochs=40)
    print(history)
