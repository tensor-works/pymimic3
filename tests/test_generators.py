import datasets
import pytest
from generators.tf2 import TFGenerator
from generators.pytorch import TorchGenerator
from generators.stream import RiverGenerator
from preprocessing.scalers import MIMICMinMaxScaler
import numpy as np
from utils.IO import *
from datasets.readers import ProcessedSetReader
from tests.settings import *
from preprocessing.imputers import PartialImputer


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_tf_generator(task_name, discretized_readers):
    tests_io(f"Test case tf2 iterative generator for task: {task_name}", level=0)

    # Prepare generator inputs
    reader = discretized_readers[task_name]
    scaler = MIMICMinMaxScaler().fit_reader(reader)

    # Bining types for LOS
    for bining in ["none", "log", "custom"]:
        # Batch sizes for dimensional robustness
        for batch_size in [1, 8, 16]:
            tests_io(f"Test case batch size: {batch_size}" + \
                    (f" and bining: {bining}" if task_name == "LOS" else ""))

            # Create generator
            generator = TFGenerator(reader=reader,
                                    scaler=scaler,
                                    batch_size=batch_size,
                                    bining=bining,
                                    shuffle=True)
            assert len(generator)
            for batch in range(len(generator)):
                # Get batch
                X, y = generator.__getitem__()
                # Check batch
                assert_batch_sanity(X, y, batch_size, bining)
                tests_io(f"Successfully tested {batch + 1} batches", flush=True)
            tests_io(f"Successfully tested {batch + 1} batches")
        if task_name != "LOS":
            break


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_torch_generator(task_name, discretized_readers):
    tests_io(f"Test case torch iterative generator for task: {task_name}", level=0)

    # Prepare generator inputs
    reader = discretized_readers[task_name]
    scaler = MIMICMinMaxScaler().fit_reader(reader)

    # Bining types for LOS
    for bining in ["none", "log", "custom"]:
        # Batch sizes for dimensional robustness
        for batch_size in [1, 8, 16]:
            tests_io(f"Test case batch size: {batch_size}" + \
                    (f" and bining: {bining}" if task_name == "LOS" else ""))
            # Create generator
            generator = TorchGenerator(reader=reader,
                                       scaler=scaler,
                                       batch_size=batch_size,
                                       bining=bining,
                                       drop_last=True,
                                       shuffle=True)
            assert len(generator)
            for batch, (X, y) in enumerate(generator):
                # Get batch
                X = X.numpy()
                y = y.numpy()
                # Check batch
                assert_batch_sanity(X, y, batch_size, bining)

                tests_io(f"Successfully tested {batch + 1} batches", flush=True)
            tests_io(f"Successfully tested {batch + 1} batches")
        if task_name != "LOS":
            break


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_river_generator(task_name, engineered_readers):
    tests_io(f"Test case river generator for task: {task_name}", level=0)

    # Prepare generator inputs
    reader = engineered_readers[task_name]
    imputer = PartialImputer().fit_reader(reader)
    scaler = MIMICMinMaxScaler(imputer=imputer).fit_reader(reader)

    # Bining types for LOS
    for bining in ["log", "custom"]:  # ["none", "log", "custom"]:
        # No Batch sizes this is a stream
        generator = RiverGenerator(reader=reader, scaler=scaler, shuffle=True, bining=bining)
        if task_name == "LOS":
            tests_io(f"Test case with bining: {bining}")
        # Create generator
        for batch, (X, y) in enumerate(generator):
            # Get batch
            X = np.fromiter(X.values(), dtype=float)
            if task_name == "PHENO" or (task_name == "LOS" and bining != "none"):
                y = np.fromiter(y.values(), dtype=float)
            assert_sample_sanity(X, y, bining)
            tests_io(f"Successfully tested {batch + 1} batches", flush=True)
        tests_io(f"Successfully tested {batch + 1} batches")
        if task_name != "LOS":
            break


def assert_batch_sanity(X: np.ndarray, y: np.ndarray, batch_size: int, bining: str):
    # The batch might be sane but I am not
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    assert X.shape[0] == batch_size
    assert X.shape[2] == 59
    assert X.dtype == np.float32
    assert y.dtype == np.int8
    assert y.shape[0] == batch_size
    if task_name in ["PHENO"]:
        assert y.shape[1] == 25
    elif task_name in ["DECOMP", "IHM"]:
        assert y.shape[1] == 1
    elif task_name in ["LOS"]:
        # Depending on the binning this changes
        if bining == "none":
            assert y.shape[1] == 1
        elif bining in ["log", "custom"]:
            assert y.shape[1] == 10


def assert_sample_sanity(X: np.ndarray, y: np.ndarray, bining: str):
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    assert len(X) == 714
    if task_name == "PHENO":
        assert len(y) == 25
    elif task_name == "LOS":
        if bining == "none":
            assert len(y) == 1
        elif bining in ["log", "custom"]:
            assert len(y) == 10
    else:
        assert isinstance(y, (float, int, bool))


if __name__ == "__main__":
    # for task_name in TASK_NAMES:
    for task_name in ["LOS"]:
        """
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    discretize=True,
                                    time_step_size=1.0,
                                    start_at_zero=True,
                                    impute_strategy='previous',
                                    task=task_name)
        # test_tf_generator(task_name, {task_name: reader})
        test_torch_generator(task_name, {task_name: reader})
        """
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    engineer=True,
                                    task=task_name)
        test_river_generator(task_name, {task_name: reader})
