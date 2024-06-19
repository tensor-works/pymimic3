import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Union, List, Dict
from utils.IO import *
from settings import *
from models.pytorch.mappings import *
from models.pytorch import AbstractTorchNetwork


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
        super().__init__(output_dim, model_path)

        self._layer_size = layer_size
        self._dropout_rate = dropout
        self._recurrent_dropout = recurrent_dropout
        self._depth = depth
        self._deep_supervision = deep_supervision

        if final_activation is None:
            if output_dim == 1:
                self._final_activation = nn.Sigmoid()
            else:
                self._final_activation = nn.Softmax(dim=-1)
        else:
            self._final_activation = activation_mapping[final_activation]

        self._output_dim = output_dim

        if isinstance(layer_size, int):
            self._hidden_sizes = [layer_size] * depth
        else:
            self._hidden_sizes = layer_size
            if depth != 1:
                warn_io("Specified hidden sizes and depth are not consistent. "
                        "Using hidden sizes and ignoring depth.")

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
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=(recurrent_dropout if i < depth - 1 else 0))

        self._dropout = nn.Dropout(dropout)
        self._output_layer = nn.Linear(input_size, self._output_dim)

    def forward(self, x, masks=None):
        if masks is not None:
            masks = masks.to(self._device)
        x = x.to(self._device)
        # Masking is not natively supported in PyTorch LSTM, assume x is already preprocessed if necessary
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x, _ = self._lstm_final(x)
        x = self._dropout(x)

        if self._deep_supervision:
            # return predictions for all timesteps
            masks.to(self._device)
            x = self._output_layer(x)
        else:
            # Only return the last prediction
            x = x[:, -1, :]
            x = self._output_layer(x)

        if self._final_activation:
            x = self._final_activation(x)
        # maks predictions
        if masks is not None:
            x = masks * x
        return x


if __name__ == "__main__":
    from tests.tsettings import *
    import datasets
    from preprocessing.scalers import MinMaxScaler
    from generators.pytorch import TorchGenerator
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=Path(SEMITEMP_DIR, "deep_supervision"),
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
                        0.2,
                        recurrent_dropout=0.,
                        output_dim=1,
                        depth=3,
                        deep_supervision=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss=criterion)
    # Example training loop
    history = model.fit(train_generator=train_generator, epochs=40)
    print(history)
