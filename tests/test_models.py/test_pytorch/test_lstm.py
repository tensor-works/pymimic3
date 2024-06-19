import datasets
import pytest
import ray
import multiprocessing as mp
from datasets.readers import ProcessedSetReader
from typing import Dict
from torch import nn
from torch import optim
from pathlib import Path
from tests.tsettings import *
from preprocessing.scalers import MinMaxScaler
from generators.pytorch import TorchGenerator
from models.pytorch.lstm import LSTMNetwork
from torcheval.metrics import BinaryAUPRC


@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_torch_lstm_with_deep_supervision(
    task_name: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    reader = discretized_readers[task_name]
    # -- Create the generator --
    scaler = MinMaxScaler().fit_reader(reader)
    train_generator = TorchGenerator(reader=reader,
                                     scaler=scaler,
                                     batch_size=2,
                                     num_cpus=0,
                                     deep_supervision=True,
                                     shuffle=True)

    # -- Create the model --
    model = LSTMNetwork(100, 59, recurrent_dropout=0., output_dim=1, depth=3, deep_supervision=True)

    # -- Compile the model --
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss=criterion,
                  metrics=["roc_auc", "pr_auc", "confusion_matrix"])

    # -- fit --
    history = model.fit(train_generator=train_generator, epochs=10)
    assert history["train_loss"][1] >= 2 * min(list(history["train_loss"].values()))

    # -- Create the dataset --
    dataset = reader.to_numpy(scaler=scaler, read_masks=True)

    # -- Create the model --
    model = LSTMNetwork(100, 59, recurrent_dropout=0., output_dim=1, depth=3, deep_supervision=True)

    # -- Compile the model --
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss=criterion,
                  metrics=["roc_auc", "pr_auc", "confusion_matrix"])

    # -- fit --
    history = model.fit([dataset["X"], dataset["M"]], dataset["yds"], batch_size=2, epochs=10)
    assert history["train_loss"][1] >= 2 * min(list(history["train_loss"].values()))


@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_torch_lstm(
    task_name: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    reader = discretized_readers[task_name]
    n_cpus = min(mp.cpu_count(), 4)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=n_cpus)
    # -- Create the generator --
    scaler = MinMaxScaler().fit_reader(reader)
    train_generator = TorchGenerator(reader=reader,
                                     scaler=scaler,
                                     batch_size=32,
                                     num_cpus=n_cpus,
                                     shuffle=True)

    # -- Create the model --
    model_path = Path(TEMP_DIR, "torch_lstm")
    model_path.mkdir(parents=True, exist_ok=True)
    model = LSTMNetwork(100, 59, recurrent_dropout=0., output_dim=1, depth=3)

    # -- Compile the model --
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss=criterion,
                  metrics=["roc_auc", "pr_auc", "confusion_matrix"])

    # -- fit --
    history = model.fit(train_generator=train_generator, epochs=10)
    ray.shutdown()
    assert history["train_loss"][1] >= 2 * min(list(history["train_loss"].values()))


if __name__ == "__main__":
    disc_reader = dict()
    for i in range(10):
        for task_name in ["DECOMP"]:
            #reader = datasets.load_data(chunksize=75836,
            #                            source_path=TEST_DATA_DEMO,
            #                            storage_path=SEMITEMP_DIR,
            #                            discretize=True,
            #                            time_step_size=1.0,
            #                            start_at_zero=True,
            #                            impute_strategy='previous',
            #                            task=task_name)

            # reader = datasets.load_data(chunksize=75836,
            #                             source_path=TEST_DATA_DEMO,
            #                             storage_path=SEMITEMP_DIR,
            #                             discretize=True,
            #                             time_step_size=1.0,
            #                             start_at_zero=True,
            #                             deep_supervision=True,
            #                             impute_strategy='previous',
            #                             task=task_name)
            reader = ProcessedSetReader(Path(SEMITEMP_DIR, "discretized", task_name))

            disc_reader[task_name] = reader
            test_torch_lstm_with_deep_supervision(task_name, disc_reader)
            test_torch_lstm(task_name, disc_reader)
