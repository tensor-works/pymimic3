import datasets
import pytest
import json
from tests.pytest_utils.models.pytorch import unroll_generator
from tests.pytest_utils.models.pytorch import assert_model_performance
from utils.IO import *
from datasets.readers import ProcessedSetReader
from typing import Dict
from torch import nn
from torch import optim
from pathlib import Path
from tests.tsettings import *
from preprocessing.scalers import MinMaxScaler
from generators.pytorch import TorchGenerator
from models.pytorch.lstm import CWLSTMNetwork
from tests.pytest_utils.decorators import retry
from tests.pytest_utils.models import assert_valid_metric
from tests.pytest_utils.models.pytorch import unroll_generator, assert_model_performance
from tests.msettings import *

# ------------------------- Target settings -------------------------
# -> target settings
TARGET_METRICS = {
    "IHM": {
        "train_loss": 1.5,
        "roc_auc": 0.85,
        "pr_auc": 0.85
    },
    "DECOMP": {
        "train_loss": 1.5,
        "roc_auc": 0.85,
        "pr_auc": 0.8,
    },
    "LOS": {
        "train_loss": 0.5,
        "cohen_kappa": 0.7,
        "custom_mae": 7
    },
    "PHENO": {
        "train_loss": 0.5,
        "micro_roc_auc": 0.7,
        "macro_roc_auc": 0.6
    }
}

TARGET_METRICS_DS = {
    "DECOMP": {
        "train_loss": 0.2,
        "roc_auc": 0.8,
        "pr_auc": 0.35,
    },
    "LOS": {
        "train_loss": 2,
        "cohen_kappa": 0.25,
        "custom_mae": 110
    }
}

TARGET_METRICS_TR = {
    "IHM": {
        "train_loss": 2.5,
        "roc_auc": 0.7,
        "pr_auc": 0.5
    },
    "PHENO": {
        "train_loss": 0.5,
        "micro_roc_auc": 0.7,
        "macro_roc_auc": 0.6
    }
}

# ------------------------- Fitting settings -------------------------
# -> fitting settings
OVERFIT_SETTINGS = {
    "IHM": {
        "epochs": 20,
        "num_samples": None
    },
    "DECOMP": {
        "epochs": 20,
        "num_samples": 400
    },
    "LOS": {
        "epochs": 20,
        "num_samples": 400
    },
    "PHENO": {
        "epochs": 20,
        "num_samples": None
    }
}

OVERFIT_SETTINGS_DS = {
    "DECOMP": {
        "epochs": 20,
        "num_samples": 400  # More samples to avoid all zeros  
    },
    "LOS": {
        "epochs": 20,
        "num_samples": 200
    }
}

OVERFIT_SETTINGS_TR = {
    "IHM": {
        "epochs": 20,
        "num_samples": None
    },
    "PHENO": {
        "epochs": 20,
        "num_samples": None
    }
}


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["IHM", "PHENO"])
@retry(3)
def test_torch_cwlstm_with_target_replication(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case torch CW-LSTM with target replication for task {task_name}", level=0)
    tests_io(f"Using {data_flavour} dataset.")

    reader = discretized_readers[task_name]
    scaler = MinMaxScaler().fit_reader(reader)

    # -- Create the model --
    # Parameters
    output_dim = OUTPUT_DIMENSIONS[task_name]
    final_activation = FINAL_ACTIVATIONS[task_name]
    model_dimensions = CHANNEL_WISE_LSTM_DS_PARAMS[task_name]["model"]
    # Obj
    model = CWLSTMNetwork(input_dim=59,
                          output_dim=output_dim,
                          final_activation=final_activation,
                          channels=reader.channels,
                          **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS_TORCH[task_name]
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
                                         n_samples=OVERFIT_SETTINGS_TR[task_name]["num_samples"],
                                         **GENERATOR_OPTIONS[task_name])
        tests_io("Succeeded in creating the generator")

        # -- Fitting the model --
        history = model.fit(generator=train_generator,
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_TR[task_name]["epochs"])

        # -- Unroll the Generator --
        X, y_true = unroll_generator(train_generator)
    elif data_flavour == "numpy":
        # -- Create the dataset --
        dataset = reader.to_numpy(scaler=scaler,
                                  n_samples=OVERFIT_SETTINGS_TR[task_name]["num_samples"],
                                  **GENERATOR_OPTIONS[task_name])

        tests_io("Succeeded in creating the numpy dataset")

        history = model.fit(dataset["X"],
                            dataset["y"],
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_TR[task_name]["epochs"])

        # -- Store dataset --
        X = dataset["X"]
        y_true = dataset["y"]

    # Instability on these is crazy but trying anyway.
    # How are you suppose to tune it with

    assert_model_performance(history, task_name, TARGET_METRICS_TR[task_name])
    assert_valid_metric(X=X, y_true=y_true, task_name=task_name, model=model, flavour="torch")
    tests_io("Succeeded in asserting model sanity")


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
@retry(3)  # Highly unstable
def test_torch_cwlstm_with_deep_supervision(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case torch CW-LSTM with deep supervision for task {task_name}", level=0)
    tests_io(f"Using {data_flavour} dataset.")

    reader = discretized_readers[task_name]
    scaler = MinMaxScaler().fit_reader(reader)

    # -- Create the model --
    # Parameters
    output_dim = OUTPUT_DIMENSIONS[task_name]
    final_activation = FINAL_ACTIVATIONS[task_name]
    model_dimensions = CHANNEL_WISE_LSTM_DS_PARAMS[task_name]["model"]
    # Obj
    model = CWLSTMNetwork(input_dim=59,
                          output_dim=output_dim,
                          final_activation=final_activation,
                          channels=reader.channels,
                          **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS_TORCH[task_name]
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

        # -- Fitting the model --
        history = model.fit(generator=train_generator,
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])

        # -- Unroll the Generator --
        X, M, y_true = unroll_generator(train_generator, deep_supervision=True)
    elif data_flavour == "numpy":
        # -- Create the dataset --
        dataset = reader.to_numpy(scaler=scaler,
                                  deep_supervision=True,
                                  n_samples=OVERFIT_SETTINGS_DS[task_name]["num_samples"],
                                  **GENERATOR_OPTIONS[task_name])

        tests_io("Succeeded in creating the numpy dataset")

        history = model.fit([dataset["X"], dataset["M"]],
                            dataset["yds"],
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])

        # -- Store dataset --
        X = dataset["X"]
        M = dataset["M"]
        y_true = dataset["yds"]

    # Instability on these is crazy but trying anyway.
    # How are you suppose to tune it with

    assert_model_performance(history, task_name, TARGET_METRICS_DS[task_name])
    assert_valid_metric(X=X,
                        y_true=y_true,
                        task_name=task_name,
                        model=model,
                        mask=M,
                        flavour="torch")
    tests_io("Succeeded in asserting model sanity")


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["IHM", "DECOMP", "LOS", "PHENO"])
@retry(3)
def test_cwtorch_lstm(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case torch CW-LSTM for task {task_name}", level=0)
    tests_io(f"Using {data_flavour} dataset.")

    reader = discretized_readers[task_name]
    scaler = MinMaxScaler().fit_reader(reader)

    # -- Create the model --
    # Parameters
    output_dim = OUTPUT_DIMENSIONS[task_name]
    final_activation = FINAL_ACTIVATIONS[task_name]
    model_dimensions = CHANNEL_WISE_LSTM_PARAMS[task_name]["model"]
    # Obj
    model = CWLSTMNetwork(input_dim=59,
                          output_dim=output_dim,
                          final_activation=final_activation,
                          channels=reader.channels,
                          **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS_TORCH[task_name]
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

        # -- Unroll the Generator --
        X, y_true = unroll_generator(train_generator)

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

        # -- Store dataset --
        X = dataset["X"]
        y_true = dataset["y"]
        y_pred = model.predict(X)
        model.evaluate(X, y_true)

    assert_model_performance(history, task_name, TARGET_METRICS[task_name])
    assert_valid_metric(X, y_true, task_name, model, flavour="torch")
    tests_io("Succeeded in asserting model sanity")


if __name__ == "__main__":
    disc_reader = dict()
    for task_name in ["IHM", "DECOMP", "LOS", "PHENO"]:
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
        reader = ProcessedSetReader(Path(SEMITEMP_DIR, "discretized", task_name))
        disc_reader[task_name] = reader
        for flavour in ["numpy", "generator"]:
            if task_name in ["IHM", "PHENO"]:
                test_torch_cwlstm_with_target_replication(task_name, flavour, disc_reader)
                ...
            if task_name in ["DECOMP", "LOS"]:
                test_torch_cwlstm_with_deep_supervision(task_name, flavour, disc_reader)
                ...
            test_cwtorch_lstm(task_name, flavour, disc_reader)
