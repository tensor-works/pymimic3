import datasets
import pytest
import ray
import multiprocessing as mp
from utils.IO import *
from datasets.readers import ProcessedSetReader
from typing import Dict
from torch import nn
from torch import optim
from pathlib import Path
from tests.tsettings import *
from preprocessing.scalers import MinMaxScaler
from generators.pytorch import TorchGenerator
from models.pytorch.lstm import LSTMNetwork


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_torch_lstm_with_deep_supervision(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(
        f"Test case torch LSTM with deep supervision using {data_flavour} "
        f"dataset for task {task_name}",
        level=0)
    reader = discretized_readers[task_name]
    scaler = MinMaxScaler().fit_reader(reader)

    # -- Create the model --
    model = LSTMNetwork(128, 59, output_dim=1)

    # -- Compile the model --
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss=criterion, metrics=["roc_auc", "pr_auc"])
    tests_io("Succeeded in creating the model")

    # -- fit --
    if data_flavour == "generator":
        # -- Create the generator --
        train_generator = TorchGenerator(reader=reader,
                                         scaler=scaler,
                                         deep_supervision=True,
                                         shuffle=True)
        tests_io("Succeeded in creating the generator")
        history = model.fit(generator=train_generator, epochs=20, batch_size=8)
    elif data_flavour == "numpy":
        # -- Create the dataset --
        dataset = reader.to_numpy(scaler=scaler, deep_supervision=True)
        history = model.fit([dataset["X"], dataset["M"]], dataset["yds"], batch_size=8, epochs=20)
        tests_io("Succeeded in creating the numpy dataset")

    # Instability on these is crazy but trying anyway.
    # How are you suppose to tune it with
    assert min(list(history["train_loss"].values())) <= 1.5, \
        f"Failed in asserting minimum loss ({min(list(history['train_loss'].values()))}) <= 1.5"
    assert max(list(history["train_metrics"]["roc_auc"].values())) >= 0.72, \
        f"Failed in asserting maximum auc ({max(list(history.history['auc']))}) >= 0.72"
    assert max(list(history["train_metrics"]["pr_auc"].values())) >= 0.12, \
        f"Failed in asserting maximum auc_1 ({max(list(history.history['auc_1']))}) >= 0.12"
    tests_io("Succeeded in asserting model sanity")


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["IHM", "DECOMP", "LOS", "PHENO"])
def test_torch_lstm(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case torch LSTM using {data_flavour} dataset for task {task_name}", level=0)
    reader = discretized_readers[task_name]
    scaler = MinMaxScaler().fit_reader(reader)

    # -- Create the model --
    model = LSTMNetwork(128, 59, output_dim=1)

    # -- Compile the model --
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss=criterion, metrics=["roc_auc", "pr_auc"])
    tests_io("Succeeded in creating the model")

    # -- fit --
    if data_flavour == "generator":
        # -- Create the generator --
        train_generator = TorchGenerator(reader=reader, scaler=scaler, shuffle=True)
        history = model.fit(generator=train_generator, epochs=5, batch_size=8)
        tests_io("Succeeded in creating the generator")
    elif data_flavour == "numpy":
        # -- Create the dataset --
        dataset = reader.to_numpy(scaler=scaler)
        history = model.fit(dataset["X"], dataset["y"], batch_size=8, epochs=5)
        tests_io("Succeeded in creating the numpy dataset")
    assert min(list(history["train_loss"].values())) <= 1.5, \
        f"Failed in asserting minimum loss ({min(list(history['train_loss'].values()))}) <= 1.5"
    assert max(list(history["train_metrics"]["roc_auc"].values())) >= 0.72, \
        f"Failed in asserting maximum auc ({max(list(history.history['auc']))}) >= 0.72"
    assert max(list(history["train_metrics"]["pr_auc"].values())) >= 0.15, \
        f"Failed in asserting maximum auc_1 ({max(list(history.history['auc_1']))}) >= 0.15"
    tests_io("Succeeded in asserting model sanity")


if __name__ == "__main__":
    import shutil
    disc_reader = dict()
    for i in range(10):
        for task_name in ["DECOMP"]:
            """
            if Path(SEMITEMP_DIR, "discretized", task_name).exists():
                shutil.rmtree(Path(SEMITEMP_DIR, "discretized", task_name))
            reader = datasets.load_data(chunksize=75836,
                                        source_path=TEST_DATA_DEMO,
                                        storage_path=SEMITEMP_DIR,
                                        discretize=True,
                                        time_step_size=1.0,
                                        start_at_zero=True,
                                        impute_strategy='previous',
                                        task=task_name)
            """
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
            for flavour in ["numpy", "generator"]:
                # test_torch_lstm_with_deep_supervision(task_name, flavour, disc_reader)
                test_torch_lstm(task_name, flavour, disc_reader)
