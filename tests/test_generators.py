import datasets
import pytest
import time
from generators.tf2 import TFGenerator
from generators.pytorch import TorchGenerator
from generators.stream import RiverGenerator
from preprocessing.scalers import MinMaxScaler
import numpy as np
from utils.IO import *
from datasets.readers import ProcessedSetReader
from tests.tsettings import *
from preprocessing.imputers import PartialImputer
from pathlib import Path


@pytest.mark.parametrize("task_name", set(TASK_NAMES) - set(["MULTI"]))
def test_tf_generator(task_name, discretized_readers, deep_supervision_readers):
    tests_io(f"Test case tf2 iterative generator for task: {task_name}", level=0)

    # Prepare generator inputs
    scaler = MinMaxScaler().fit_reader(reader)

    # Bining types for LOS
    for bining in ["none", "log", "custom"]:
        # Batch sizes for dimensional robustness
        for batch_size in [1, 8, 16]:
            for ds_mode in ["standard", "deep_supervision", "target_replication"]:
                if ds_mode in ["standard", "target_replication"]:
                    cur_reader = discretized_readers[task_name]
                elif ds_mode == "deep_supervision":
                    if task_name in ["PHENO", "IHM"]:
                        continue
                    cur_reader = deep_supervision_readers[task_name]
                tests_io(f"Test case batch size: {batch_size}" + \
                        (f" and bining: {bining}" if task_name == "LOS" else "") + \
                        (f" and deep supervision" if ds_mode == "deep_supervision" else ""))

                # Create generator
                generator = TFGenerator(reader=cur_reader,
                                        scaler=scaler,
                                        batch_size=batch_size,
                                        deep_supervision=(ds_mode == "deep_supervision"),
                                        target_replication=(ds_mode == "target_replication"),
                                        bining=bining,
                                        shuffle=True)
                assert len(generator)
                for batch in range(len(generator)):
                    # Get batch
                    X, y = generator.__getitem__()
                    if ds_mode == "deep_supervision":
                        X, M = X
                    # Check batch
                    else:
                        M = None
                    assert_batch_sanity(X=X,
                                        y=y,
                                        batch_size=batch_size,
                                        bining=bining,
                                        M=M,
                                        target_repl=(ds_mode == "target_replication"))
                    tests_io(f"Successfully tested {batch + 1} batches", flush=True)
                tests_io(f"Successfully tested {batch + 1} batches")
        if task_name != "LOS":
            break


@pytest.mark.parametrize("task_name", set(TASK_NAMES) - set(["MULTI"]))
def test_torch_generator(task_name, discretized_readers, deep_supervision_readers):
    tests_io(f"Test case torch iterative generator for task: {task_name}", level=0)

    # Prepare generator inputs
    scaler = MinMaxScaler().fit_reader(reader)
    reader = discretized_readers[task_name]

    # Bining types for LOS
    for bining in ["none", "log", "custom"]:
        # Batch sizes for dimensional robustness
        for batch_size in [1, 8, 16]:
            for ds_mode in ["standard", "deep_supervision"]:
                # if ds_mode in ["standard", "target_replication"]:
                # elif ds_mode == "deep_supervision":
                #     if task_name in ["PHENO", "IHM"]:
                #         continue
                #    cur_reader = deep_supervision_readers[task_name]
                tests_io(f"Test case batch size: {batch_size}" + \
                        (f" and bining: {bining}" if task_name == "LOS" else "") + \
                        (f" and deep supervision" if ds_mode == "deep_supervision" else ""))
                # Create generator
                generator = TorchGenerator(reader=reader,
                                           scaler=scaler,
                                           batch_size=batch_size,
                                           deep_supervision=(ds_mode == "deep_supervision"),
                                           target_replication=(ds_mode == "target_replication"),
                                           bining=bining,
                                           drop_last=True,
                                           shuffle=True)
                assert len(generator)
                start = time.time()
                for batch, (X, y) in enumerate(generator):
                    # Get batch
                    if ds_mode == "deep_supervision":
                        X, M = X
                        X = X.numpy()
                        M = M.numpy()
                    elif ds_mode == "standard":
                        X = X.numpy()
                        M = None
                    y = y.numpy()
                    # Check batch
                    assert_batch_sanity(X=X,
                                        y=y,
                                        batch_size=batch_size,
                                        bining=bining,
                                        M=M,
                                        target_repl=(ds_mode == "target_replication"))
                    tests_io(f"Successfully tested {batch + 1} batches", flush=True)
                tests_io(f"Successfully tested {batch + 1} batches")
                end = time.time()
                elapsed_time = end - start
                minutes = int(elapsed_time // 60)
                seconds = elapsed_time % 60

                tests_io(f"Time enrolling the generator was: {minutes} min, {seconds:.2f} sec")
        if task_name != "LOS":
            break


@pytest.mark.parametrize("task_name", set(TASK_NAMES) - set(["MULTI"]))
def test_river_generator(task_name, engineered_readers):
    tests_io(f"Test case river generator for task: {task_name}", level=0)

    # Prepare generator inputs
    reader = engineered_readers[task_name]
    imputer = PartialImputer().fit_reader(reader)
    scaler = MinMaxScaler(imputer=imputer).fit_reader(reader)

    # Bining types for LOS
    for bining in ["none", "log", "custom"]:
        # No Batch sizes this is a stream
        generator = RiverGenerator(reader=reader, scaler=scaler, shuffle=True, bining=bining)
        if task_name == "LOS":
            tests_io(f"Test case with bining: {bining}")
        # Create generator
        for batch, (X, y) in enumerate(generator):
            # Get batch
            X = np.fromiter(X.values(), dtype=float)
            # No trace of one-hot encoding these in the original code base
            if task_name == "PHENO":  # or (task_name == "LOS" and bining != "none"):
                y = np.fromiter(y.values(), dtype=float)
            assert_sample_sanity(X, y, bining)
            tests_io(f"Successfully tested {batch + 1} samples", flush=True)
        tests_io(f"Successfully tested {batch + 1} batches")
        if task_name != "LOS":
            break


def assert_batch_sanity(X: np.ndarray,
                        y: np.ndarray,
                        batch_size: int,
                        bining: str,
                        one_hot: bool = False,
                        M=None):
    # The batch might be sane but I am not
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    assert np.all((X >= 0) & (X <= 1))
    assert X.shape[0] == batch_size
    assert X.shape[2] == 59
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    assert y.shape[0] == batch_size
    if M is not None:
        assert M.shape == y.shape
        assert M.dtype == y.dtype
        content_index = 2
    else:
        content_index = 1
    if task_name in ["PHENO"]:
        assert y.shape[content_index] == 25
    elif task_name in ["DECOMP"]:
        assert y.shape[content_index] == 1
    elif task_name in ["IHM"]:
        assert y.shape[content_index] == 1
    elif task_name in ["LOS"]:
        # Depending on the binning this changes
        if bining == "none":
            assert y.shape[content_index] == 1
        elif bining in ["log", "custom"] and one_hot:
            assert y.shape[content_index] == 10


def assert_sample_sanity(X: np.ndarray, y: np.ndarray, bining: str, one_hot: bool = False):
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    assert len(X) == 714
    if task_name == "PHENO":
        assert len(y) == 25
    elif task_name == "LOS" and one_hot:
        if bining == "none":
            assert len(y) == 1
        elif bining in ["log", "custom"]:
            assert len(y) == 10
    else:
        assert isinstance(y, (float, int, bool))


if __name__ == "__main__":
    for task_name in TASK_NAMES:
        if task_name == "MULTI":
            continue
        st_reader = datasets.load_data(chunksize=75836,
                                       source_path=TEST_DATA_DEMO,
                                       storage_path=SEMITEMP_DIR,
                                       discretize=True,
                                       time_step_size=1.0,
                                       start_at_zero=True,
                                       impute_strategy='previous',
                                       task=task_name)
        if task_name in ["DECOMP", "LOS"]:
            ds_reader = datasets.load_data(chunksize=75836,
                                           source_path=TEST_DATA_DEMO,
                                           storage_path=Path(SEMITEMP_DIR, "deep_supervision"),
                                           discretize=True,
                                           time_step_size=1.0,
                                           start_at_zero=True,
                                           deep_supervision=True,
                                           impute_strategy='previous',
                                           task=task_name)
        test_torch_generator(task_name, {task_name: st_reader}, {task_name: ds_reader})
        test_tf_generator(task_name, {task_name: st_reader}, {task_name: ds_reader})
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    engineer=True,
                                    task=task_name)
        test_river_generator(task_name, {task_name: reader})
