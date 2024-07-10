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

TARGET_METRICS = {
    "IHM": {
        "train_loss": 1.5,
        "roc_auc": 0.75,
        "pr_auc": 0.15
    },
    "DECOMP": {
        "train_loss": 1.5,
        "roc_auc": 0.72,
        "pr_auc": 0.15,
    },
    "LOS": {
        "train_loss": 1.8,
        "cohen_kappa": 0.8,
        "custom_mae": 50
    },
    "PHENO": {
        "train_loss": 1.5,
        "micro_roc_auc": 0.8,
        "macro_roc_auc": 0.8
    }
}


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
                                         shuffle=True,
                                         n_samples=OVERFIT_SETTINGS_DS[task_name]["num_samples"],
                                         **GENERATOR_OPTIONS[task_name])
        tests_io("Succeeded in creating the generator")
        history = model.fit(generator=train_generator,
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])

    elif data_flavour == "numpy":
        # -- Create the dataset --
        dataset = reader.to_numpy(scaler=scaler,
                                  deep_supervision=True,
                                  n_samples=OVERFIT_SETTINGS_DS[task_name]["num_samples"],
                                  **GENERATOR_OPTIONS[task_name])
        history = model.fit([dataset["X"], dataset["M"]],
                            dataset["yds"],
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])
        tests_io("Succeeded in creating the numpy dataset")

    # Instability on these is crazy but trying anyway.
    # How are you suppose to tune it with

    assert_model_performance(history, task_name)
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
                                         n_samples=OVERFIT_SETTINGS[task_name]["num_samples"],
                                         shuffle=True,
                                         **GENERATOR_OPTIONS[task_name])
        tests_io("Succeeded in creating the generator")

        # -- Fitting the model --
        history = model.fit(generator=train_generator,
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS[task_name]["epochs"])

    elif data_flavour == "numpy":
        # -- Task specific settings --
        # -- Create the dataset --
        tests_io("Loading the numpy dataset...", end="\r")
        dataset = reader.to_numpy(scaler=scaler,
                                  n_samples=OVERFIT_SETTINGS[task_name]["num_samples"],
                                  **GENERATOR_OPTIONS[task_name])
        tests_io("Done loading the numpy dataset")

        # -- Fitting the model --
        history = model.fit(dataset["X"],
                            dataset["y"],
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS[task_name]["epochs"])

    assert_model_performance(history, task_name)
    tests_io("Succeeded in asserting model sanity")


def assert_model_performance(history, task):
    target_metrics = TARGET_METRICS[task]

    for metric, target_value in target_metrics.items():
        if metric == "train_loss":
            actual_value = min(list(history[metric].values()))
            comparison = actual_value <= target_value
        else:
            actual_value = max(list(history["train_metrics"][metric].values()))
            comparison = actual_value >= target_value if "mae" not in metric else actual_value <= target_value

        assert comparison, \
            (f"Failed in asserting {metric} ({actual_value}) "
             f"{'<=' if 'loss' in metric or 'mae' in metric  else '>='} {target_value} for task {task}")


if __name__ == "__main__":
    import shutil
    disc_reader = dict()
    for task_name in ["DECOMP", "LOS", "PHENO"]:
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
        if task_name in ["DECOMP", "LOS"]:
            reader = datasets.load_data(chunksize=75836,
                                        source_path=TEST_DATA_DEMO,
                                        storage_path=SEMITEMP_DIR,
                                        discretize=True,
                                        time_step_size=1.0,
                                        start_at_zero=True,
                                        deep_supervision=True,
                                        impute_strategy='previous',
                                        task=task_name)
        """
        reader = ProcessedSetReader(Path(SEMITEMP_DIR, "discretized", task_name))
        disc_reader[task_name] = reader
        for flavour in ["numpy", "generator"]:  # ["generator"]:  #
            if task_name in ["DECOMP", "LOS"]:
                # test_torch_lstm_with_deep_supervision(task_name, flavour, disc_reader)
                pass
            test_torch_lstm(task_name, flavour, disc_reader)
