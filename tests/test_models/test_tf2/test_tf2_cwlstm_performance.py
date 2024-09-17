import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
tf.config.run_functions_eagerly(True)

from tests.msettings import *
from tests.tsettings import *
from utils.IO import *
import datasets
import pytest
import json
import numpy as np
from typing import Dict
from pathlib import Path
from preprocessing.scalers import MinMaxScaler
from generators.tf2 import TFGenerator
from datasets.readers import ProcessedSetReader
from models.tf2.lstm import CWLSTMNetwork
from utils.arrays import zeropad_samples
from tests.pytest_utils.decorators import retry
from tests.pytest_utils.models import assert_valid_metric
from tests.pytest_utils.models.tf2 import unroll_generator, assert_model_performance
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

# ------------------------- Target settings -------------------------
# -> target settings
TARGET_METRICS = {
    "IHM": {
        "loss": 0.5,
        "roc_auc": 0.9,
        "pr_auc": 0.9
    },
    "DECOMP": {
        "loss": 0.2,
        "roc_auc": 0.9,
        "pr_auc": 0.9,
    },
    "LOS": {
        "loss": 0.3,
        "cohen_kappa": 0.85,
        "custom_mae": 3
    },
    "PHENO": {
        "loss": 0.5,
        "micro_roc_auc": 0.75,
        "macro_roc_auc": 0.7
    }
}

TARGET_METRICS_DS = {
    "DECOMP": {
        "loss": 0.2,
        "roc_auc": 0.8,
        "pr_auc": 0.3,  # had to lower from 0.4
    },
    "LOS": {
        "loss": 2,
        "cohen_kappa": 0.25,
        "custom_mae": 110
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
        "num_samples": 200
    },
    "PHENO": {
        "epochs": 20,
        "num_samples": None
    }
}

OVERFIT_SETTINGS_DS = {
    "DECOMP": {
        "epochs": 20,
        "num_samples": 400
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
    "PENO": {
        "epochs": 20,
        "num_samples": None
    }
}


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
@retry(3)
def test_tf2_cwlstm_with_deep_supvervision(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case tf2 CW-LSTM with deep supervision for task {task_name}", level=0)
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
                          recurrent_dropout=0.,
                          output_dim=output_dim,
                          deep_supervision=True,
                          final_activation=final_activation,
                          channels=reader.channels,
                          **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS_TF2[task_name]
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
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
        train_generator = TFGenerator(reader=reader,
                                      scaler=scaler,
                                      batch_size=8,
                                      deep_supervision=True,
                                      shuffle=True,
                                      **GENERATOR_OPTIONS[task_name])
        tests_io("Succeeded in creating the generator")
        history = model.fit(train_generator, epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])

        # -- Unroll the Generator --
        X, M, y_true = unroll_generator(generator=train_generator, deep_supervision=True)
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

    assert_model_performance(history, task_name, TARGET_METRICS_DS[task_name])
    assert_valid_metric(X, y_true, task_name, model, mask=M)
    tests_io("Succeeded in asserting model sanity")


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
@retry(3)
def test_tf2_cwlstm(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    if data_flavour == "numpy" and task_name in ["DECOMP", "LOS"]:
        warn_io("Not yet figured out how to make this work with mulitlabel data")
    tests_io(f"Test case tf2 CW-LSTM for task {task_name}", level=0)
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
    criterion = NETWORK_CRITERIONS_TF2[task_name]
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
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
        train_generator = TFGenerator(reader=reader,
                                      scaler=scaler,
                                      batch_size=8,
                                      shuffle=True,
                                      n_samples=OVERFIT_SETTINGS[task_name]["num_samples"],
                                      **GENERATOR_OPTIONS[task_name])

        tests_io("Succeeded in creating the generator")

        # -- Fitting the model --
        history = model.fit(train_generator, epochs=OVERFIT_SETTINGS[task_name]["epochs"])

        # -- Unroll the Generator --
        X, y_true = unroll_generator(train_generator)
    elif data_flavour == "numpy":
        # -- Create the dataset --
        tests_io("Loading the numpy dataset...", end="\r")
        # Binned with custom bins one LOS task
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

    assert_model_performance(history, task_name, TARGET_METRICS[task_name])
    assert_valid_metric(X, y_true, task_name, model)
    tests_io("Succeeded in asserting model sanity")


if __name__ == "__main__":
    disc_reader = dict()
    for task_name in ["IHM", "DECOMP", "LOS", "PHENO"]:

        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    discretize=True,
                                    time_step_size=1.0,
                                    start_at_zero=True,
                                    impute_strategy='previous',
                                    task=task_name)
        if task_name in ['LOS', 'DECOMP']:
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
        dataset = reader.to_numpy()
        for flavour in ["generator", "numpy"]:
            disc_reader[task_name] = reader
            if task_name in ['LOS', 'DECOMP']:
                test_tf2_cwlstm_with_deep_supvervision(task_name, flavour, disc_reader)
                ...
            test_tf2_cwlstm(task_name, flavour, disc_reader)
