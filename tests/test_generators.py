import datasets
import pytest
from generators.tf2 import TFGenerator
from generators.pytorch import TorchGenerator
from generators.scikit import ScikitGenerator, ScikitDataset
from preprocessing.scalers import MIMICMinMaxScaler
import numpy as np
from utils.IO import *
from datasets.readers import ProcessedSetReader
from tests.settings import *
from preprocessing.imputers import PartialImputer


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_tf_generator(task_name, discretized_readers):
    tests_io(f"Test case iterative generator for task: {task_name}", level=0)
    reader = discretized_readers[task_name]
    scaler = MIMICMinMaxScaler().fit_reader(reader)
    for batch_size in [1, 8, 16]:
        tests_io(f"Test case batch size: {batch_size}")
        generator = TFGenerator(reader=reader, scaler=scaler, batch_size=batch_size, shuffle=True)
        assert len(generator)
        for batch in range(len(generator)):
            X, y = generator.__getitem__()
            assert not np.isnan(X).any()
            assert not np.isnan(y).any()
            assert X.shape[0] == batch_size
            assert X.shape[2] == 59
            assert y.shape[0] == batch_size
            assert X.dtype == np.float32
            assert y.dtype == np.float32
            tests_io(f"Successfully tested {batch + 1} batches", flush=True)
    tests_io(f"Successfully tested {batch + 1} batches")


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_torch_generator(task_name, discretized_readers):
    tests_io(f"Test case iterative generator for task: {task_name}", level=0)
    reader = discretized_readers[task_name]
    scaler = MIMICMinMaxScaler().fit_reader(reader)
    for batch_size in [1, 8, 16]:
        tests_io(f"Test case batch size: {batch_size}")
        generator = TorchGenerator(reader=reader,
                                   scaler=scaler,
                                   batch_size=batch_size,
                                   drop_last=True,
                                   shuffle=True)
        assert len(generator)
        for batch, (X, y) in enumerate(generator):
            X = X.numpy()
            y = y.numpy()
            assert not np.isnan(X).any()
            assert not np.isnan(y).any()
            assert X.shape[0] == batch_size
            assert X.shape[2] == 59
            assert X.dtype == np.float32
            assert y.dtype == np.float32
            tests_io(f"Successfully tested {batch + 1} batches", flush=True)
        tests_io(f"Successfully tested {batch + 1} batches")


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_scikit_generator(task_name, engineered_readers):
    tests_io(f"Test case iterative generator for task: {task_name}", level=0)
    reader = engineered_readers[task_name]
    imputer = PartialImputer().fit_reader(reader)
    scaler = MIMICMinMaxScaler(imputer=imputer).fit_reader(reader)

    for batch_size in [1, 8, 16]:
        tests_io(f"Test case batch size: {batch_size}")
        generator = ScikitGenerator(reader=reader,
                                    scaler=scaler,
                                    batch_size=batch_size,
                                    drop_last=True,
                                    shuffle=True)
        assert len(generator)
        for batch, (X, y) in enumerate(generator):
            assert not np.isnan(X).any()
            assert not np.isnan(y).any()
            assert X.shape[0] == batch_size
            assert X.shape[1] == 714
            assert X.dtype == np.float32
            assert y.dtype == np.float32
            tests_io(f"Successfully tested {batch + 1} batches", flush=True)
        tests_io(f"Successfully tested {batch + 1} batches")


if __name__ == "__main__":
    for task_name in TASK_NAMES:
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    discretize=True,
                                    time_step_size=1.0,
                                    start_at_zero=True,
                                    impute_strategy='previous',
                                    task=task_name)
        # test_tf_generator(task_name, {task_name: reader})
        # test_torch_generator(task_name, {task_name: reader})
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    engineer=True,
                                    task=task_name)
        test_scikit_generator(task_name, {task_name: reader})
