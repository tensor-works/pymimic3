import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Dict
from utils.IO import *
from settings import *
from collections import OrderedDict
from models.pytorch.mappings import *
from models.pytorch import AbstractTorchNetwork
from utils.arrays import isiterable


class LSTMNetwork(AbstractTorchNetwork):

    def __init__(self,
                 layer_size: Union[List[int], int],
                 input_dim: int,
                 dropout: float = 0.,
                 recurrent_dropout: float = 0.,
                 final_activation: str = None,
                 output_dim: int = 1,
                 depth: int = 1,
                 target_repl_coef: float = 0.,
                 model_path: Path = None):
        super().__init__(final_activation=final_activation,
                         output_dim=output_dim,
                         model_path=model_path,
                         target_repl_coef=target_repl_coef)

        self._layer_size = layer_size
        self._dropout_rate = dropout
        self._recurrent_dropout = recurrent_dropout
        self._depth = depth
        self._output_dim = output_dim

        if isinstance(layer_size, int):
            self._hidden_sizes = [layer_size] * (depth - 1)
            last_layer_size = layer_size
        elif isiterable(layer_size):
            self._hidden_sizes = layer_size[:-1]
            last_layer_size = layer_size[-1]
            if depth != 1:
                warn_io("Specified hidden sizes and depth are not consistent. "
                        "Using hidden sizes and ignoring depth.")
        else:
            raise ValueError("Layer size must be an integer or a list of integers.")

        self.lstm_layers = nn.ModuleList()
        input_size = input_dim
        for i, hidden_size in enumerate(self._hidden_sizes):
            self.lstm_layers.append(
                nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=(recurrent_dropout if i < depth - 1 else 0)))
            input_size = hidden_size

        self._lstm_final = nn.LSTM(input_size=input_size,
                                   hidden_size=last_layer_size,
                                   num_layers=1,
                                   batch_first=True)

        self._dropout = nn.Dropout(dropout)
        # Output layer
        self._output_layer = nn.Linear(last_layer_size, self._output_dim)

        # Initialize the network with same methods as in TF2
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('tanh'))
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data, gain=nn.init.calculate_gain('sigmoid'))
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

    def forward(self, x, masks=None) -> torch.Tensor:
        masking_flag = masks is not None
        if masking_flag:
            masks = masks.to(self._device)
        x = x.to(self._device)

        # Masking is not natively supported in PyTorch LSTM, assume x is already preprocessed if necessary
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x, _ = self._lstm_final(x)

        # Case 1: deep supervision
        if masking_flag:
            # Apply the linear layer to each LSTM output at each timestep (ts)
            B, T, hidden_size = x.shape
            x = x.view(B * T, hidden_size)
            x = self._output_layer(x)
            x = x.view(B, T, -1)

        # Case 2: standard LSTM or target replication
        else:
            # Apply linear layer only to the last output of the LSTM
            x = x[:, -1, :]
            x = self._output_layer(x)
            x = x.unsqueeze(1)

        # Apply final activation if specified
        if self._final_activation and self._apply_activation:
            x = self._final_activation(x)

        return x


class CWLSTMNetwork(AbstractTorchNetwork):

    def __init__(self,
                 clayer_size: Union[List[int], int],
                 layer_size: Union[List[int], int],
                 input_dim: int,
                 channels: List[str],
                 dropout: float = 0.,
                 recurrent_dropout: float = 0.,
                 final_activation: str = None,
                 output_dim: int = 1,
                 depth: int = 1,
                 target_repl_coef: float = 0.,
                 model_path: Path = None):
        super(CWLSTMNetwork, self).__init__(final_activation=final_activation,
                                            output_dim=output_dim,
                                            model_path=model_path,
                                            target_repl_coef=target_repl_coef)

        self._input_dim = input_dim
        self._layer_size = layer_size
        self._clayer_size = clayer_size
        self._depth = depth
        self._dropout = dropout
        self._recurrent_dropout = recurrent_dropout
        self._output_dim = output_dim

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

        # Block 1: Channel wise LSTM layers
        self._lstm_per_channel = nn.ModuleDict()
        for channel_names, channel_indices in self._channels.items():
            channel_lstms = nn.ModuleList()
            if isiterable(clayer_size):
                for i, size in enumerate(clayer_size):
                    lstm = nn.LSTM(input_size=len(channel_indices) if i == 0 else clayer_size[i -
                                                                                              1],
                                   hidden_size=size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=recurrent_dropout if i < len(clayer_size) - 1 else 0)
                    channel_lstms.append(lstm)
            else:
                for i in range(depth):
                    lstm = nn.LSTM(input_size=len(channel_indices) if i == 0 else clayer_size,
                                   hidden_size=clayer_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=recurrent_dropout if i < depth - 1 else 0)
                    channel_lstms.append(lstm)
            self._lstm_per_channel[channel_names] = channel_lstms

        # Block 2: Concatenated LSTM layers
        self._concat_lstm_layers = nn.ModuleList()
        if isiterable(layer_size):
            input_size = clayer_size[-1] * len(self._channels)
            for i, size in enumerate(layer_size[-1]):

                lstm = nn.LSTM(input_size=input_size if i == 0 else layer_size[i - 1],
                               hidden_size=size,
                               num_layers=1,
                               batch_first=True,
                               dropout=recurrent_dropout if i < depth - 1 else 0)
                self._concat_lstm_layers.append(lstm)
            final_input_size = layer_size[-2] if len(layer_size) > 1 else input_size
        else:
            input_size = clayer_size * len(self._channels)
            for i in range(depth - 1):
                lstm = nn.LSTM(input_size=input_size if i == 0 else layer_size,
                               hidden_size=layer_size,
                               num_layers=1,
                               batch_first=True,
                               dropout=dropout if i < depth - 1 else 0)
                self._concat_lstm_layers.append(lstm)
            final_input_size = layer_size if depth > 1 else input_size

        # Final concatenated LSTM layers
        lstm = nn.LSTM(input_size=final_input_size,
                       hidden_size=layer_size,
                       num_layers=1,
                       batch_first=True)
        self._concat_lstm_layers.append(lstm)

        self._dropout_layer = nn.Dropout(dropout)

        # Output layer
        self._output_layer = nn.Linear(layer_size, output_dim)

        # Initialize the network with same methods as in TF2
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('tanh'))
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data, gain=nn.init.calculate_gain('sigmoid'))
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

    def forward(self, x, masks=None) -> torch.Tensor:
        masking_flag = masks is not None
        if masking_flag:
            masks = masks.to(self._device)

        x = x.to(self._device)

        # Block 1: Channel wise forward (clayers)
        channel_outputs = list()
        for channel_name, channel_lstm in self._lstm_per_channel.items():
            cw_x = x[:, :, self._channels[channel_name]]
            for lstm in channel_lstm:
                cw_x, _ = lstm(cw_x)
            channel_outputs.append(cw_x)

        # Concatenate channel outputs
        x = torch.cat(channel_outputs, dim=2)

        # Block 2: Concatenated forward
        for lstm in self._concat_lstm_layers:
            x, _ = lstm(x)

        # Case 1: deep supervision
        if masking_flag:
            # Apply the linear layer to each LSTM output at each timestep (ts)
            B, T, hidden_size = x.shape
            x = x.view(B * T, hidden_size)
            x = self._output_layer(x)
            x = x.view(B, T, -1)

        # Case 2: standard LSTM or target replication
        else:
            # Apply linear layer only to the last output of the LSTM
            x = x[:, -1, :]
            x = self._output_layer(x)
            x = x.unsqueeze(1)

        # Apply final activation if specified
        if self._final_activation and self._apply_activation:
            x = self._final_activation(x)

        return x


if __name__ == "__main__":
    from tests.tsettings import *
    import datasets
    from preprocessing.scalers import MinMaxScaler
    from generators.pytorch import TorchGenerator
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
    train_generator = TorchGenerator(reader=reader,
                                     scaler=scaler,
                                     deep_supervision=True,
                                     shuffle=True)
    # val_generator = TorchGenerator(reader=reader.val,
    #                                scaler=scaler,
    #                                batch_size=2,
    #                                deep_supervision=True,
    #                                shuffle=True)

    model_path = Path(TEMP_DIR, "torch_lstm")
    model_path.mkdir(parents=True, exist_ok=True)
    model = LSTMNetwork(1000, 59, output_dim=1, depth=1, final_activation="softmax")
    import torch
    import numpy as np
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss=criterion, metrics=["pr_auc", "roc_auc"])
    # Example training loop
    history = model.fit(generator=train_generator,
                        validation_data=train_generator,
                        epochs=1,
                        batch_size=8)
    model.evaluate(train_generator)
    x, y = np.random.uniform(0, 1, (200, 100, 59)), np.random.choice([0, 1], (200, 1, 1))
    model.fit(x, y, epochs=1, batch_size=8, validation_data=(x, y))
    model.evaluate(x, y)
    print()
