import datasets
import pytest
import time
import ray
import numpy as np
from generators.tf2 import TFGenerator
from generators.pytorch import TorchGenerator
from generators.stream import RiverGenerator
from preprocessing.scalers import MinMaxScaler
from utils.IO import *
from datasets.readers import ProcessedSetReader
from tests.tsettings import *
from preprocessing.imputers import PartialImputer
from pathlib import Path
from typing import Dict
from pathos import multiprocessing as mp


@pytest.mark.parametrize("mode", ["deep_supervision", "standard"])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
@pytest.mark.parametrize("multiprocessed", [True, False])
def test_tf_generators_with_ds(task_name: str, batch_size: int, mode: str, multiprocessed: bool,
                               discretized_readers: Dict[str, ProcessedSetReader]):
    tests_io(f"Test case tf2 generator for task: {task_name}", level=0)
    reader = discretized_readers[task_name]

    if multiprocessed:
        if ray.is_initialized():
            ray.shutdown()
        n_cpus = min(mp.cpu_count(), 4)
        ray.init(num_cpus=n_cpus)

    # Test reader supervision
    X, y, m = reader.random_samples(1, read_masks=True)
    assert len(X), "Deep supervision has not been created correctly. Missing samples"
    assert len(y), "Deep supervision has not been created correctly. Missing labels"
    assert len(m), "Deep supervision has not been created correctly. Missing mask"

    # Prepare generator inputs
    scaler = MinMaxScaler().fit_reader(reader)

    # Bining types for LOS
    for bining in ["log", "custom", "none"]:
        tests_io(f"Test case batch size: {batch_size}" + \
                (f"\nbining: {bining}" if task_name == "LOS" else "") + \
                (f"\ndeep supervision" if mode == "deep_supervision" else "") + \
                (f"\ntarget replication" if mode == "target_replication" else ""))

        # Create generator
        generator = TFGenerator(reader=reader,
                                scaler=scaler,
                                batch_size=batch_size,
                                num_cpus=n_cpus if multiprocessed else 0,
                                deep_supervision=(mode == "deep_supervision"),
                                target_replication=(mode == "target_replication"),
                                bining=bining,
                                shuffle=True)
        assert len(generator)

        for batch in range(len(generator)):
            # Get batch
            X, y = generator.__getitem__()
            if mode == "deep_supervision":
                X, M = X
            # Check batch
            else:
                M = None
            assert_batch_sanity(X=X,
                                y=y,
                                task_name=task_name,
                                batch_size=batch_size,
                                bining=bining,
                                M=M,
                                target_repl=False)
            tests_io(f"Successfully tested {batch + 1} batches", flush=True)
        tests_io(f"Successfully tested {batch + 1} batches\n")
        if task_name != "LOS":
            break
    if multiprocessed:
        ray.shutdown()


@pytest.mark.parametrize("mode", ["target_replication", "standard"])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("task_name", ["IHM", "PHENO"])
@pytest.mark.parametrize("multiprocessed", [True, False])
def test_tf_generators_with_tr(task_name: str, batch_size: int, mode: str, multiprocessed: bool,
                               discretized_readers):
    tests_io(f"Test case tf2 generator for task: {task_name}", level=0)
    reader = discretized_readers[task_name]

    if multiprocessed:
        if ray.is_initialized():
            ray.shutdown()
        n_cpus = min(mp.cpu_count(), 4)
        ray.init(num_cpus=n_cpus)
    # Prepare generator inputs
    scaler = MinMaxScaler().fit_reader(reader)

    # Bining types for LOS
    tests_io(f"Test case batch size: {batch_size}" + \
            (f"\ndeep supervision" if mode == "deep_supervision" else "") + \
            (f"\ntarget replication" if mode == "target_replication" else ""))

    # Create generator
    generator = TFGenerator(reader=reader,
                            scaler=scaler,
                            batch_size=batch_size,
                            num_cpus=n_cpus if multiprocessed else 0,
                            deep_supervision=(mode == "deep_supervision"),
                            target_replication=(mode == "target_replication"),
                            shuffle=True)
    assert len(generator)
    for batch in range(len(generator)):
        # Get batch
        X, y = generator.__getitem__()
        assert_batch_sanity(X=X,
                            y=y,
                            task_name=task_name,
                            batch_size=batch_size,
                            target_repl=(mode == "target_replication"))
        tests_io(f"Successfully tested {batch + 1} batches", flush=True)
    tests_io(f"Successfully tested {batch + 1} batches\n")
    if multiprocessed:
        ray.shutdown()


@pytest.mark.parametrize("mode", ["deep_supervision", "standard"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
@pytest.mark.parametrize("multiprocessed", [True, False])
def test_torch_generators_with_ds(task_name: str, mode: str, multiprocessed: bool,
                                  discretized_readers: Dict[str, ProcessedSetReader]):
    tests_io(f"Test case torch generator for task: {task_name}", level=0)
    reader = discretized_readers[task_name]

    if multiprocessed:
        if ray.is_initialized():
            ray.shutdown()
        n_cpus = min(mp.cpu_count(), 4)
        ray.init(num_cpus=n_cpus)
    # Prepare generator inputs

    # Test reader supervision
    X, y, m = reader.random_samples(1, read_masks=True)
    assert len(X), "Deep supervision has not been created correctly. Missing samples"
    assert len(y), "Deep supervision has not been created correctly. Missing labels"
    assert len(m), "Deep supervision has not been created correctly. Missing mask"

    scaler = MinMaxScaler().fit_reader(reader)

    # Bining types for LOS
    for bining in ["none", "log", "custom"]:
        tests_io(f"Test case batch size: 1" + \
                (f"\nbining: {bining}" if task_name == "LOS" else "") + \
                (f"\ndeep supervision" if mode == "deep_supervision" else "") + \
                (f"\ntarget replication" if mode == "target_replication" else ""))

        # Create generator
        generator = TorchGenerator(reader=reader,
                                   scaler=scaler,
                                   num_cpus=n_cpus if multiprocessed else 0,
                                   deep_supervision=(mode == "deep_supervision"),
                                   target_replication=(mode == "target_replication"),
                                   bining=bining,
                                   drop_last=True,
                                   shuffle=True)
        assert len(generator)
        start = time.time()
        for batch, (X, y) in enumerate(generator):
            # Get batch
            if mode == "deep_supervision":
                X, M = X
                X = X.numpy()
                M = M.numpy()
            elif mode in ["standard", "target_replication"]:
                X = X.numpy()
                M = None
            y = y.numpy()
            # Check batch
            assert_batch_sanity(X=X,
                                y=y,
                                task_name=task_name,
                                batch_size=1,
                                bining=bining,
                                M=M,
                                target_repl=False)
            tests_io(f"Successfully tested {batch + 1} batches", flush=True)
        tests_io(f"Successfully tested {batch + 1} batches\n")
        end = time.time()
        elapsed_time = end - start
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        tests_io(f"Time enrolling the generator was: {minutes} min, {seconds:.2f} sec")
        if task_name != "LOS":
            break
    if multiprocessed:
        ray.shutdown()


@pytest.mark.parametrize("mode", ["target_replication", "standard"])
@pytest.mark.parametrize("task_name", ["IHM", "PHENO"])
@pytest.mark.parametrize("multiprocessed", [True, False])
def test_torch_generators_with_tr(task_name: str, mode: str, multiprocessed: bool,
                                  discretized_readers: Dict[str, ProcessedSetReader]):
    tests_io(f"Test case torch generator for task: {task_name}", level=0)
    reader = discretized_readers[task_name]

    if multiprocessed:
        n_cpus = min(mp.cpu_count(), 4)
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=n_cpus)
    # Prepare generator inputs
    scaler = MinMaxScaler().fit_reader(reader)

    # Bining types for LOS
    tests_io(f"Test case batch size: 1" + \
            (f"\ndeep supervision" if mode == "deep_supervision" else "") + \
            (f"\ntarget replication" if mode == "target_replication" else ""))

    # Create generator
    generator = TorchGenerator(reader=reader,
                               scaler=scaler,
                               num_cpus=n_cpus if multiprocessed else 0,
                               deep_supervision=False,
                               target_replication=(mode == "target_replication"),
                               drop_last=True,
                               shuffle=True)
    assert len(generator)
    start = time.time()
    print("Generator dim is: ", len(generator))
    for batch, (X, y) in enumerate(generator):
        # Get batch
        X = X.numpy()
        y = y.numpy()
        # Check batch
        assert_batch_sanity(X=X,
                            y=y,
                            task_name=task_name,
                            batch_size=1,
                            target_repl=(mode == "target_replication"))
        tests_io(f"Successfully tested {batch + 1} batches", flush=True)
    tests_io(f"Successfully tested {batch + 1} batches\n")

    end = time.time()
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    tests_io(f"Time enrolling the generator was: {minutes} min, {seconds:.2f} sec")
    if multiprocessed:
        ray.shutdown()


@pytest.mark.parametrize("task_name", set(TASK_NAMES) - set(["MULTI"]))
@pytest.mark.parametrize("multiprocessed", [True, False])
def test_river_generator(task_name: str, multiprocessed: bool,
                         engineered_readers: ProcessedSetReader):
    tests_io(f"Test case river generator for task: {task_name}", level=0)
    reader = engineered_readers[task_name]

    if multiprocessed:
        n_cpus = min(mp.cpu_count(), 4)
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=n_cpus)
    # Prepare generator inputs
    imputer = PartialImputer().fit_reader(reader)
    scaler = MinMaxScaler(imputer=imputer).fit_reader(reader)

    # Bining types for LOS
    for bining in ["none", "log", "custom"]:
        # No Batch sizes this is a stream
        generator = RiverGenerator(reader=reader,
                                   scaler=scaler,
                                   num_cpus=n_cpus if multiprocessed else 0,
                                   shuffle=True,
                                   bining=bining)
        if task_name == "LOS":
            tests_io(f"Test case with bining: {bining}")
        # Create generator
        for batch, (X, y) in enumerate(generator):
            # Get batch
            X = np.fromiter(X.values(), dtype=float)
            # No trace of one-hot encoding these in the original code base
            if task_name == "PHENO":  # or (task_name == "LOS" and bining != "none"):
                y = np.fromiter(y.values(), dtype=float)
            assert_sample_sanity(X, y, task_name, bining)
            tests_io(f"Successfully tested {batch + 1} samples", flush=True)
        tests_io(f"Successfully tested {batch + 1} batches\n")
        if task_name != "LOS":
            break
    if multiprocessed:
        ray.shutdown()


def assert_batch_sanity(X: np.ndarray,
                        y: np.ndarray,
                        task_name: str,
                        batch_size: int,
                        bining: str = "none",
                        one_hot: bool = False,
                        M: np.ndarray = None,
                        target_repl: bool = False):
    # The batch might be sane but I am not
    # B = batch_size, T = time_steps, F = features, N = number of classes
    # (B, T, F) for X ]
    # (B, N) for y without deep supervision or target replication
    # (B, T, N) for y with deep supervision or target replication
    # (B, T) for M with deep supervision

    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    assert np.all((X >= 0 - 1e-6) & (X <= 1 + 1e-6))
    # X[0] = B
    assert X.shape[0] == batch_size
    # X[2] = F
    assert X.shape[2] == 59
    # y[0] = B
    assert y.shape[0] == batch_size
    assert X.dtype == np.float32
    if bining in ['log', 'custom']:
        assert y.dtype == np.int64, f"Erroneous target dtype! Expected int64 but found {y.dtype}."
    else:
        assert y.dtype == np.float32, f"Erroneous target dtype! Expected int32 but found {y.dtype}."
    if M is not None:
        # Y[1] = T && X[1] = T
        assert y.shape[1] == X.shape[1]
        # M[0] = B && M[1] = T
        assert M.shape == y.shape
        assert M.dtype == bool
    elif target_repl:
        # Y[1] = T && X[1] = T
        assert y.shape[1] == X.shape[1]

    # y[2] = N
    if task_name in ["PHENO"]:
        assert y.shape[2] == 25
    elif task_name in ["DECOMP"]:
        assert y.shape[2] == 1
    elif task_name in ["IHM"]:
        assert y.shape[2] == 1
    elif task_name in ["LOS"]:
        # Depending on the binning this changes
        if bining == "none":
            assert y.shape[2] == 1
        elif bining in ["log", "custom"] and one_hot:
            assert y.shape[2] == 10


def assert_sample_sanity(X: np.ndarray,
                         y: np.ndarray,
                         task_name: str,
                         bining: str = "none",
                         one_hot: bool = False):
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    # F = features, N = number of classes
    # There is not batch mode for a lot of river models
    # (F) for X
    # (N) for y
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
    """
    if SEMITEMP_DIR.is_dir():
        import shutil
        shutil.rmtree(SEMITEMP_DIR)
    """
    for task_name in TASK_NAMES:

        if not Path(SEMITEMP_DIR, "discretized", task_name).is_dir():
            st_reader = datasets.load_data(chunksize=75836,
                                           source_path=TEST_DATA_DEMO,
                                           storage_path=SEMITEMP_DIR,
                                           discretize=True,
                                           time_step_size=1.0,
                                           start_at_zero=True,
                                           impute_strategy='previous',
                                           task=task_name)
        else:
            st_reader = ProcessedSetReader(Path(SEMITEMP_DIR, "discretized", task_name))
        if task_name in ["DECOMP", "LOS"]:
            ds_reader = datasets.load_data(chunksize=75836,
                                           source_path=TEST_DATA_DEMO,
                                           storage_path=SEMITEMP_DIR,
                                           discretize=True,
                                           time_step_size=1.0,
                                           start_at_zero=True,
                                           deep_supervision=True,
                                           impute_strategy='previous',
                                           task=task_name)

        if task_name == "MULTI":
            continue

        if not Path(SEMITEMP_DIR, "engineered", task_name).is_dir():
            reader = datasets.load_data(chunksize=75836,
                                        source_path=TEST_DATA_DEMO,
                                        storage_path=SEMITEMP_DIR,
                                        engineer=True,
                                        task=task_name)
        else:
            reader = ProcessedSetReader(Path(SEMITEMP_DIR, "engineered", task_name))

        # The ds reader fixture is not accessed but ensured the set is also created
        # deep supervision

        if task_name in ["DECOMP", "LOS"]:
            for mode in ["deep_supervision", "standard"]:
                for batch_size in [1, 16]:
                    test_tf_generators_with_ds(task_name=task_name,
                                               batch_size=batch_size,
                                               mode=mode,
                                               multiprocessed=False,
                                               discretized_readers={task_name: st_reader})
                test_torch_generators_with_ds(task_name=task_name,
                                              mode=mode,
                                              multiprocessed=False,
                                              discretized_readers={task_name: st_reader})

        if task_name in ["IHM", "PHENO"]:
            for mode in ["target_replication", "standard"]:
                test_torch_generators_with_tr(task_name=task_name,
                                              mode=mode,
                                              multiprocessed=False,
                                              discretized_readers={task_name: st_reader})
                for batch_size in [1, 16]:
                    test_tf_generators_with_tr(task_name=task_name,
                                               batch_size=batch_size,
                                               mode=mode,
                                               multiprocessed=False,
                                               discretized_readers={task_name: st_reader})

        test_river_generator(task_name=task_name,
                             multiprocessed=False,
                             engineered_readers={task_name: reader})
