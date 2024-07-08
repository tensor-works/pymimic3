import datasets
import pytest
from tests.tsettings import *
from datasets.readers import ProcessedSetReader
from typing import Dict
from pathlib import Path
from utils.IO import *
from tests.tsettings import *
from preprocessing.scalers import MinMaxScaler
from generators.tf2 import TFGenerator
from models.tf2.lstm import LSTMNetwork
from tests.msettings import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


@pytest.mark.parametrize("data_flavour", ["generator", "numpy"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_tf2_lstm_with_deep_supvervision(
    task_name: str,
    data_flavour: str,
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(
        f"Test case tf2 LSTM with deep supervision using {data_flavour} "
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
                        recurrent_dropout=0.,
                        output_dim=output_dim,
                        deep_supervision=True,
                        final_activation=final_activation,
                        **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS[task_name]
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=criterion, metrics=NETWORK_METRICS[task_name])
    tests_io("Succeeded in creating the model")

    # -- fit --
    if data_flavour == "generator":
        # -- Create the generator --
        train_generator = TFGenerator(reader=reader,
                                      scaler=scaler,
                                      batch_size=8,
                                      deep_supervision=True,
                                      shuffle=True)
        tests_io("Succeeded in creating the generator")
        history = model.fit(train_generator, epochs=10)
    elif data_flavour == "numpy":
        # -- Create the dataset --
        dataset = reader.to_numpy(scaler=scaler, deep_supervision=True)
        tests_io("Succeeded in creating the numpy dataset")
        history = model.fit([dataset["X"], dataset["M"]], dataset["yds"], batch_size=8, epochs=10)
    assert min(list(history.history["loss"])) <= 1.5, \
        f"Failed in asserting minimum loss ({min(list(history.history['loss']))}) <= 1.5"
    assert max(list(history.history["auc"])) >= 0.8, \
        f"Failed in asserting maximum auc ({max(list(history.history['auc']))}) >= 0.8"
    assert max(list(history.history["auc_1"])) >= 0.2, \
        f"Failed in asserting maximum auc_1 ({max(list(history.history['auc_1']))}) >= 0.45"
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
    tests_io(f"Test case tf2 LSTM using {data_flavour} dataset for task {task_name}", level=0)
    reader = discretized_readers[task_name]

    scaler = MinMaxScaler().fit_reader(reader)

    # -- Create the model --
    # Parameters
    output_dim = OUTPUT_DIMENSIONS[task_name]
    final_activation = FINAL_ACTIVATIONS[task_name]
    model_dimensions = STANDARD_LSTM_PARAMS[task_name]["model"]
    # Obj
    model = LSTMNetwork(input_dim=59,
                        recurrent_dropout=0.,
                        output_dim=output_dim,
                        final_activation=final_activation,
                        **model_dimensions)

    # -- Compile the model --
    criterion = NETWORK_CRITERIONS[task_name]
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=criterion, metrics=NETWORK_METRICS[task_name])
    tests_io("Succeeded in creating the model")

    # -- fit --
    if data_flavour == "generator":
        # -- Create the generator --
        train_generator = TFGenerator(reader=reader, scaler=scaler, batch_size=8, shuffle=True)
        tests_io("Succeeded in creating the generator")

        # -- Fitting the model --
        history = model.fit(train_generator, epochs=5)

    elif data_flavour == "numpy":
        # -- Create the dataset --
        tests_io("Loading the numpy dataset...", end="\r")
        # Binned with custom bins one LOS task
        dataset = reader.to_numpy(scaler=scaler,
                                  one_hot=task_name == "LOS",
                                  **GENERATOR_OPTIONS[task_name])
        tests_io("Done loading the numpy dataset")

        # -- Fitting the model --
        history = model.fit(dataset["X"], dataset["y"], batch_size=8, epochs=10)

    assert min(list(history.history["loss"])) <= 1.3, \
        f"Failed in asserting minimum loss ({min(list(history.history['loss']))}) <= 1.5"
    assert max(list(history.history["auc"])) >= 0.8, \
        f"Failed in asserting maximum auc ({max(list(history.history['auc']))}) >= 0.8"
    assert max(list(history.history["auc_1"])) >= 0.4, \
        f"Failed in asserting maximum auc_1 ({max(list(history.history['auc_1']))}) >= 0.45"
    tests_io("Succeeded in asserting model sanity")


if __name__ == "__main__":
    disc_reader = dict()
    for i in range(10):
        for task_name in ["LOS"]:
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
            dataset = reader.to_numpy()
            for flavour in ["numpy", "generator"]:
                disc_reader[task_name] = reader
                test_tf2_lstm(task_name, flavour, disc_reader)
                test_tf2_lstm_with_deep_supvervision(task_name, flavour, disc_reader)
