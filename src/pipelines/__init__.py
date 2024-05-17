from typing import Union
from pathlib import Path
import datasets
import pytest
from model.callbacks import HistoryCheckpoint

from generators.tf2 import TFGenerator
from generators.torch import TorchGenerator
from preprocessing.scalers import AbstractScaler, MIMICMinMaxScaler, MIMICStandardScaler, MIMICMaxAbsScaler, MIMICRobustScaler
import numpy as np
from managers import HistoryManager
from managers import CheckpointManager
from utils.IO import *
from datasets.readers import ProcessedSetReader
from tests.settings import *
from datasets.readers import ProcessedSetReader, SplitSetReader
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class TFPipeline:

    def __init__(
        self,
        storage_path: Path,
        reader: Union[ProcessedSetReader, SplitSetReader],
        model,
        generator_options: dict,
        model_options: dict = {},
        scaler_options: dict = {},
        compile_options: dict = {},
        data_split_options: dict = {},
        scaler: AbstractScaler = None,
        scaler_type: str = "minmax",
    ):
        self._storage_path = storage_path
        self._reader = reader
        self._model = model
        self._generator_options = generator_options
        self._data_split_options = data_split_options
        self._split_names = ["train"]

        self._init_scaler(scaler_type=scaler_type,
                          scaler_options=scaler_options,
                          scaler=scaler,
                          reader=reader)

        self._init_generators(generator_options=generator_options)

        self._init_model(model=model, model_options=model_options, compiler_options=compile_options)

    def _init_scaler(self, scaler_type: str, scaler_options: dict, scaler: AbstractScaler,
                     reader: Union[ProcessedSetReader, SplitSetReader]):
        if scaler is not None:
            self._scaler = scaler
            return
        if isinstance(self._reader, ProcessedSetReader):
            reader = self._reader
        elif isinstance(self._reader, SplitSetReader):
            reader = self._reader.train
        else:
            raise ValueError(
                f"Invalid reader type: {type(self._reader)}. Must be either ProcessedSetReader or SplitSetReader."
            )
        if not scaler_type in ["minmax", "standard"]:
            raise ValueError(
                f"Invalid scaler type: {scaler_type}. Must be either 'minmax' or 'standard'.")
        if scaler_type == "minmax":
            self._scaler = MIMICMinMaxScaler(storage_path=self._storage_path,
                                             **scaler_options).fit_reader(reader)
        elif scaler_type == "standard":
            self._scaler = MIMICStandardScaler(storage_path=self._storage_path,
                                               **scaler_options).fit_reader(reader)
        elif scaler_type == "maxabs":
            self._scaler = MIMICMaxAbsScaler(storage_path=self._storage_path,
                                             **scaler_options).fit_reader(reader)
        elif scaler_type == "robust":
            self._scaler = MIMICRobustScaler(storage_path=self._storage_path,
                                             **scaler_options).fit_reader(reader)

    def _init_generators(self, generator_options: dict):
        if isinstance(self._reader, ProcessedSetReader):
            self._train_generator = TFGenerator(reader=self._reader,
                                                scaler=self._scaler,
                                                **generator_options)
        elif isinstance(self._reader, SplitSetReader):
            self._train_generator = TFGenerator(reader=self._reader.train,
                                                scaler=self._scaler,
                                                **generator_options)
            if "val" in self._reader.split_names:
                self._split_names.append("val")
                self._val_generator = TFGenerator(reader=self._reader.val,
                                                  scaler=self._scaler,
                                                  **generator_options)
            else:
                self._val_generator = None

            if "test" in self._reader.split_names:
                self._split_names.append("test")
                self._test_generator = TFGenerator(reader=self._reader.test,
                                                   scaler=self._scaler,
                                                   **generator_options)

    def _init_model(self, model, model_options, compiler_options):
        if isinstance(model, type):
            self._model = model(**model_options)

        if model.optimizer is None:
            self._model.compile(**compiler_options)

    def _init_callbacks(self,
                        epochs: int = None,
                        patience: int = None,
                        restore_best_weights=True,
                        save_weights_only: bool = False,
                        save_best_only: bool = True):
        self._es_callback = EarlyStopping(patience=patience,
                                          restore_best_weights=restore_best_weights)

        self._cp_callback = ModelCheckpoint(filepath=Path(self.directories["model_path"],
                                                          "cp-{epoch:04d}.ckpt"),
                                            save_weights_only=save_weights_only,
                                            save_best_only=save_best_only,
                                            verbose=0)

        self._hist_callback = HistoryCheckpoint(Path(self._result_path, "history.json"))

        self._hist_manager = HistoryManager(self._result_path)

        self._manager = CheckpointManager(self._result_path,
                                          epochs,
                                          custom_objects=self.custom_objects)

    def _init_result_path(self, result_name: str, result_path: Path):
        if result_path is not None:
            self._result_path = Path(result_path, result_name)
        else:
            self._result_path = Path(self._storage_path, "results", result_name)

    def fit(self,
            result_name: str,
            result_path: Path = None,
            epochs: int = None,
            patience: int = None,
            callbacks: list = [],
            restore_best_weights=True,
            save_weights_only: bool = False,
            class_weight: dict = None,
            sample_weight: dict = None,
            save_best_only: bool = True,
            validation_freq: int = 1):

        self._init_result_path(result_name, result_path)
        self._init_callbacks(epochs=epochs,
                             patience=patience,
                             restore_best_weights=restore_best_weights,
                             save_weights_only=save_weights_only,
                             save_best_only=save_best_only)

        self._model.fit(self._train_generator,
                        validation_data=self._val_generator,
                        epochs=epochs,
                        steps_per_epoch=self._train_generator.steps,
                        callbacks=callbacks +
                        [self._es_callback, self._cp_callback, self._hist_callback],
                        class_weight=class_weight,
                        sample_weight=sample_weight,
                        initial_epoch=self._manager.latest_epoch(),
                        validation_steps=self._val_generator.steps,
                        validation_freq=validation_freq)

        self._hist_manager.finished()
        _, best_epoch = self._hist_manager.best
        self._manager.clean_directory(best_epoch)

        return self._model


if __name__ == "__main__":
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task="IHM")

    from model.tf2.lstm import LSTM
    from tests.settings import *
    model = LSTM(10,
                 0.2,
                 59,
                 bidirectional=False,
                 recurrent_dropout=0.,
                 task=None,
                 target_repl=False,
                 output_dim=1,
                 depth=1)

    pipe = TFPipeline(storage_path=Path(TEMP_DIR, "tf_pipeline"),
                      reader=reader,
                      model=model,
                      generator_options={
                          "batch_size": 16,
                          "shuffle": True
                      }).fit(epochs=10)
    """
    model = LSTM(10,
                 0.2,
                 59,
                 bidirectional=False,
                 recurrent_dropout=0.,
                 task=None,
                 target_repl=False,
                 output_dim=1,
                 depth=1)

    generator = TFGenerator(reader=reader, scaler=scaler, batch_size=2, shuffle=True)
    # Create and compile your model

    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Fit the model
    batch_size = 32
    steps_per_epoch = 100

    model.fit(x=generator, steps_per_epoch=steps_per_epoch, epochs=10)
    """
    scaler = MinMaxScaler().fit_reader(reader)
    generator = TorchGenerator(reader=reader, scaler=scaler, batch_size=2, shuffle=True)

    import torch
    import torch.nn as nn
    import torch.optim as optim

    class TimeSeriesModel(nn.Module):

        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(TimeSeriesModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # Initialize hidden state and cell state
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))

            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out

    # Define the model parameters
    input_size = 59
    hidden_size = 128
    num_layers = 2
    output_size = 1

    # Initialize the model, loss function, and optimizer
    model = TimeSeriesModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Example training loop
    num_epochs = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(generator):
            # Move tensors to the configured device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(generator)}], Loss: {loss.item():.4f}'
                )

# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.summary()
# model.fit(generator, epochs=1, steps_per_epoch=generator._steps, verbose=1)
