import datasets
import pytest
import json
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
from tests.msettings import *


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
    # Parameters
    output_dim = OUTPUT_DIMENSIONS[task_name]
    final_activation = FINAL_ACTIVATIONS[task_name]
    model_dimensions = STANDARD_LSTM_DS_PARAMS[task_name]["model"]
    # Obj
    model = LSTMNetwork(input_dim=59,
                        output_dim=output_dim,
                        final_activation=final_activation,
                        **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS[task_name]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss=criterion, metrics=NETWORK_METRICS[task_name])

    # Let them know
    tests_io(f"Succeeded in creating the model with:\n"
             f"output dim: {output_dim}\n"
             f"final_activation: {final_activation}\n"
             f"model_dimension: {json.dumps(model_dimensions, indent=4)}\n"
             f"criterion: {criterion}\n"
             f"optimizer: Adam, lr=0.001")

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
    # Parameters
    output_dim = OUTPUT_DIMENSIONS[task_name]
    final_activation = FINAL_ACTIVATIONS[task_name]
    model_dimensions = STANDARD_LSTM_PARAMS[task_name]["model"]
    # Obj
    model = LSTMNetwork(input_dim=59,
                        output_dim=output_dim,
                        final_activation=final_activation,
                        **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS[task_name]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss=criterion, metrics=NETWORK_METRICS[task_name])

    # Let them know
    tests_io(f"Succeeded in creating the model with:\n"
             f"output dim: {output_dim}\n"
             f"final_activation: {final_activation}\n"
             f"model_dimension: {json.dumps(model_dimensions, indent=4)}\n"
             f"criterion: {criterion}\n"
             f"optimizer: Adam, lr=0.001")

    # -- fit --
    if data_flavour == "generator":
        # -- Create the generator --
        train_generator = TorchGenerator(reader=reader,
                                         scaler=scaler,
                                         shuffle=True,
                                         **GENERATOR_OPTIONS[task_name])
        tests_io("Succeeded in creating the generator")

        # -- Fitting the model --
        history = model.fit(generator=train_generator, epochs=5, batch_size=8)

    elif data_flavour == "numpy":
        # -- Create the dataset --
        tests_io("Loading the numpy dataset...", end="\r")
        dataset = reader.to_numpy(scaler=scaler, **GENERATOR_OPTIONS[task_name])
        tests_io("Done loading the numpy dataset")

        # -- Fitting the model --
        if task_name == "IHM":
            epochs = 20
        elif task_name == "PHENO":
            epochs = 20
        else:
            epochs = 5
        history = model.fit(dataset["X"], dataset["y"], batch_size=8, epochs=epochs)

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
        for task_name in ["DECOMP"]:  # ["PHENO"]:  # ["IHM", "DECOMP", "PHENO", "LOS"]:
            """
            if Path(SEMITEMP_DIR, "discretized", task_name).exists():
                shutil.rmtree(Path(SEMITEMP_DIR, "discretized", task_name))
            """
            reader = datasets.load_data(chunksize=75836,
                                        source_path=TEST_DATA_DEMO,
                                        storage_path=SEMITEMP_DIR,
                                        discretize=True,
                                        time_step_size=1.0,
                                        start_at_zero=True,
                                        impute_strategy='previous',
                                        task=task_name)
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
            for flavour in ["numpy"]:  #, "generator"]:
                # test_torch_lstm_with_deep_supervision(task_name, flavour, disc_reader)
                test_torch_lstm(task_name, flavour, disc_reader)
