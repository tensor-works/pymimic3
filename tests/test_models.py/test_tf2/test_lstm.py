import datasets
import pytest
import ray
import multiprocessing as mp
from datasets.readers import ProcessedSetReader
from typing import Dict
from pathlib import Path
from tests.tsettings import *
from preprocessing.scalers import MinMaxScaler
from generators.tf2 import TFGenerator
from models.tf2.lstm import LSTMNetwork
from tensorflow.keras.optimizers import Adam


@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_tf2_lstm_with_deep_supvervision(task_name: str,
                                         discretized_readers: Dict[str, ProcessedSetReader]):
    reader = discretized_readers[task_name]

    # -- Create the generator --
    scaler = MinMaxScaler().fit_reader(reader)
    train_generator = TFGenerator(reader=reader,
                                  scaler=scaler,
                                  batch_size=32,
                                  deep_supervision=True,
                                  shuffle=True)

    # -- Create the model --
    model = LSTMNetwork(500,
                        59,
                        recurrent_dropout=0.,
                        output_dim=1,
                        depth=2,
                        deep_supervision=True,
                        final_activation='sigmoid')

    # -- Compile the model --
    optimizer = Adam(learning_rate=0.000001, clipvalue=1.0)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy')  # , metrics=["roc_auc", "roc_pr"])

    # -- fit --
    history = model.fit(train_generator, epochs=10)
    assert history["train_loss"][1] >= 1.5 * min(list(history["train_loss"].values()))

    # -- Create the dataset --
    dataset = reader.to_numpy(scaler=scaler, read_masks=True)

    # -- Create the model --
    model = LSTMNetwork(1000,
                        59,
                        recurrent_dropout=0.,
                        output_dim=1,
                        depth=3,
                        deep_supervision=True,
                        final_activation='sigmoid')

    # -- Compile the model --
    optimizer = Adam(learning_rate=0.000001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    # -- fit --
    history = model.fit([dataset["X"], dataset["M"]], dataset["yds"], epochs=100)
    assert history["train_loss"][1] >= 2 * min(list(history["train_loss"].values()))


if __name__ == "__main__":
    disc_reader = dict()
    for i in range(10):
        for task_name in ["DECOMP"]:
            # reader = datasets.load_data(chunksize=75836,
            #                             source_path=TEST_DATA_DEMO,
            #                             storage_path=SEMITEMP_DIR,
            #                             discretize=True,
            #                             time_step_size=1.0,
            #                             start_at_zero=True,
            #                             impute_strategy='previous',
            #                             task=task_name)

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
            test_tf2_lstm_with_deep_supvervision(task_name, disc_reader)
            # test_tf_lstm(task_name, disc_reader)
