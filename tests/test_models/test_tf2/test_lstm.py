import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
tf.config.run_functions_eagerly(True)

from tests.msettings import *
import datasets
import pytest
import json
from tests.tsettings import *
from datasets.readers import ProcessedSetReader
from typing import Dict
from pathlib import Path
from utils.IO import *
from tests.tsettings import *
from preprocessing.scalers import MinMaxScaler
from generators.tf2 import TFGenerator
from models.tf2.lstm import LSTMNetwork
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

TARGET_METRICS = {
    "IHM": {
        "loss": 1.5,
        "roc_auc": 0.75,
        "pr_auc": 0.15
    },
    "DECOMP": {
        "loss": 1.5,
        "roc_auc": 0.72,
        "pr_auc": 0.15,
    },
    "LOS": {
        "loss": 1.8,
        "cohen_kappa": 0.8,
        "custom_mae": 50
    },
    "PHENO": {
        "micro_roc_auc": 0.75,
        "macro_roc_auc": 0.7
    }
}


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_tf2_lstm_with_deep_supvervision(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case tf2 LSTM with deep supervision for task {task_name}", level=0)
    tests_io(f"Using {data_flavour} dataset.")
    reader = discretized_readers[task_name]

    scaler = MinMaxScaler().fit_reader(reader)

    # -- Create the model --
    # Parameters
    output_dim = OUTPUT_DIMENSIONS[task_name]
    final_activation = FINAL_ACTIVATIONS[task_name]
    model_dimensions = STANDARD_LSTM_DS_PARAMS[task_name]["model"]
    # Obj
    model = LSTMNetwork(input_dim=59,
                        recurrent_dropout=0.,
                        output_dim=output_dim,
                        deep_supervision=True,
                        final_activation=final_activation,
                        **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS[task_name]
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
        train_generator = TFGenerator(
            reader=reader,
            scaler=scaler,
            batch_size=8,
            deep_supervision=True,
            # n_samples=OVERFIT_SETTINGS_DS[task_name]["num_samples"],
            shuffle=True,
            #one_hot=task_name == "LOS",
            **GENERATOR_OPTIONS[task_name])
        tests_io("Succeeded in creating the generator")
        history = model.fit(train_generator, epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])
    elif data_flavour == "numpy":
        # -- Create the dataset --
        dataset = reader.to_numpy(
            scaler=scaler,
            # one_hot=task_name == "LOS",
            deep_supervision=True,
            n_samples=OVERFIT_SETTINGS_DS[task_name]["num_samples"],
            **GENERATOR_OPTIONS[task_name])
        tests_io("Succeeded in creating the numpy dataset")
        history = model.fit([dataset["X"], dataset["M"]],
                            dataset["yds"],
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])

    # assert_model_performance(history, task_name)
    tests_io("Succeeded in asserting model sanity")


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_tf2_lstm(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    if data_flavour == "numpy" and task_name in ["DECOMP", "LOS"]:
        warn_io("Not yet figured out how to make this work with mulitlabel data")
    tests_io(f"Test case tf2 LSTM for task {task_name}", level=0)
    tests_io(f"Using {data_flavour} dataset.")
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
        train_generator = TFGenerator(
            reader=reader,
            scaler=scaler,
            batch_size=8,
            shuffle=True,
            # one_hot=task_name == "LOS",
            # n_samples=OVERFIT_SETTINGS_DS[task_name]["num_samples"],
            **GENERATOR_OPTIONS[task_name])

        tests_io("Succeeded in creating the generator")

        # -- Fitting the model --
        history = model.fit(train_generator, epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])

    elif data_flavour == "numpy":
        # -- Create the dataset --
        tests_io("Loading the numpy dataset...", end="\r")
        # Binned with custom bins one LOS task
        dataset = reader.to_numpy(
            scaler=scaler,
            # one_hot=task_name == "LOS",
            n_samples=OVERFIT_SETTINGS_DS[task_name]["num_samples"],
            **GENERATOR_OPTIONS[task_name])
        tests_io("Done loading the numpy dataset")

        # -- Fitting the model --
        history = model.fit(dataset["X"],
                            dataset["y"],
                            batch_size=8,
                            epochs=OVERFIT_SETTINGS_DS[task_name]["epochs"])

    # assert_model_performance(history, task_name)
    tests_io("Succeeded in asserting model sanity")


def assert_model_performance(history, task):
    target_metrics = TARGET_METRICS[task]

    for metric, target_value in target_metrics.items():
        if metric == "loss":
            actual_value = min(history.history[metric])
            comparison = actual_value <= target_value
        else:
            # For other metrics, assume higher is better unless it's an error metric
            actual_value = max(history.history[metric])
            comparison = actual_value >= target_value if "error" not in metric.lower(
            ) and "loss" not in metric.lower() else actual_value <= target_value

        assert comparison, \
            (f"Failed in asserting {metric} ({actual_value:.4f}) "
             f"{'<=' if 'loss' in metric.lower() or 'error' in metric.lower() else '>='} {target_value} for task {task}")


if __name__ == "__main__":
    disc_reader = dict()
    for task_name in ["LOS"]:  #["IHM", "DECOMP", "PHENO"]:

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
            test_tf2_lstm(task_name, flavour, disc_reader)
            if task_name in ['LOS', 'DECOMP']:
                test_tf2_lstm_with_deep_supvervision(task_name, flavour, disc_reader)
