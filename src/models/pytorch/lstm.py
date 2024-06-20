import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Union, List, Dict
from utils.IO import *
from settings import *
from models.pytorch.mappings import *
from models.pytorch import AbstractTorchNetwork
from utils import is_iterable
import torch.nn as nn


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Compute TimeDistributed layer
        batch_size = x.size(0)
        time_steps = x.size(1)
        remaining_dims = x.size()[2:]

        # Reshape input tensor for module
        if self.batch_first:
            x = x.contiguous().view(batch_size * time_steps, *remaining_dims)
        else:
            x = x.contiguous().view(time_steps, batch_size, *remaining_dims).transpose(0, 1)

        # Apply module and reshape output
        y = self.module(x)
        if self.batch_first:
            y = y.view(batch_size, time_steps, *y.size()[1:])
        else:
            y = y.view(time_steps, batch_size, *y.size()[1:]).transpose(0, 1)

        return y


class TimeDistributedDense(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.contiguous().view(-1, x.size(-1))  # Combine batch_size and seq_len
        y = self.dense(x_reshaped)
        y = y.view(batch_size, seq_len, -1)  # Restore batch_size and seq_len
        return y


class LSTMNetwork(AbstractTorchNetwork):

    def __init__(self,
                 layer_size: Union[List[int], int],
                 input_dim: int,
                 dropout: float = 0.,
                 deep_supervision: bool = False,
                 recurrent_dropout: float = 0.,
                 final_activation: str = None,
                 output_dim: int = 1,
                 depth: int = 1,
                 model_path: Path = None):
        super().__init__(final_activation, output_dim, model_path)

        self._layer_size = layer_size
        self._dropout_rate = dropout
        self._recurrent_dropout = recurrent_dropout
        self._depth = depth
        self._deep_supervision = deep_supervision
        self._output_dim = output_dim

        if isinstance(layer_size, int):
            self._hidden_sizes = [layer_size] * (depth - 1)
            last_layer_size = layer_size
        elif is_iterable(layer_size):
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
        self._output_layer = nn.Linear(
            last_layer_size,
            self._output_dim)  # TimeDistributedDense(last_layer_size, self._output_dim)
        #

        for lstm in [self._lstm_final, self._output_layer]:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x, masks=None):
        if masks is not None:
            masks = masks.to(self._device)
        x = x.to(self._device)

        # Masking is not natively supported in PyTorch LSTM, assume x is already preprocessed if necessary
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x, _ = self._lstm_final(x)

        if self._deep_supervision:
            outputs = list()
            # Apply the linear to teach timestep
            for ts in range(x.shape[1]):
                outputs.append(self._output_layer(x[:, ts, :]))
            # Cat to vector
            x = torch.cat(outputs, dim=1)
            if len(x.shape) < 3:
                x = x.unsqueeze(-1)

        else:
            # Only return the last prediction
            x = x[:, -1, :]
            x = self._output_layer(x)

        if self._final_activation:
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
                                     batch_size=8,
                                     deep_supervision=True,
                                     shuffle=True)
    # val_generator = TorchGenerator(reader=reader.val,
    #                                scaler=scaler,
    #                                batch_size=2,
    #                                deep_supervision=True,
    #                                shuffle=True)

    model_path = Path(TEMP_DIR, "torch_lstm")
    model_path.mkdir(parents=True, exist_ok=True)
    model = LSTMNetwork(1000,
                        59,
                        output_dim=1,
                        depth=1,
                        final_activation="softmax",
                        deep_supervision=True)
    import torch
    criterion = nn.BCELoss(torch.tensor([4.0], dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    model.compile(optimizer=optimizer, loss=criterion, metrics=["pr_auc", "roc_auc"])
    # Example training loop
    history = model.fit(train_generator=train_generator, epochs=40)
    print(history)
