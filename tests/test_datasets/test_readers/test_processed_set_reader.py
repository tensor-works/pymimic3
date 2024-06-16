import datasets
import pytest
import pandas as pd
import numpy as np
from copy import deepcopy
from utils.IO import *
from tests.tsettings import *
from settings import *
from typing import Dict
from datasets.readers import ProcessedSetReader
from datasets.mimic_utils import upper_case_column_names

timeseries_label_cols = {
    "IHM":
        "Y",
    "DECOMP":
        "Y",
    "LOS":
        "Y",
    "PHENO": [
        'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13',
        'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22', 'y23', 'y24'
    ],
    "MULTI": [
        'PHENO_LABELS', 'IHM_POS', 'IHM_MASK', 'IHM_LABEL', 'DECOMP_MASKS', 'DECOMP_LABELS',
        'LOS_MASKS', 'LOS_LABELS', 'LOS_VALUE'
    ]
}

DTYPES = {
    column_name.upper(): np.float64 if dtype == "float64" else object
    for column_name, dtype in DATASET_SETTINGS["timeseries"]["dtype"].items()
}

# TODO! Warnings because we do not have timestamps anymore


@pytest.mark.parametrize("task_name", TASK_NAMES)
@pytest.mark.parametrize("reader_flavour", ["discretized", "engineered", "preprocessed"])
def test_read_sample(
    task_name: str,
    reader_flavour: str,
    preprocessed_readers: Dict[str, ProcessedSetReader],
    engineered_readers: Dict[str, ProcessedSetReader],
    discretized_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case read sample for task {task_name}", level=0)
    if reader_flavour == "preprocessed":
        reader = preprocessed_readers[task_name]
    elif reader_flavour == "discretized":
        reader = discretized_readers[task_name]
    elif reader_flavour == "engineered":
        reader = engineered_readers[task_name]
    # 10017: Single stay
    # 40124: Multiple stays

    # Testing without reading ids and timestamps
    check_sample(reader.read_sample(10017),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    check_sample(reader.read_sample(40124),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} without ids and timestamps passed")

    # Testing with reading ids and without timestamps
    check_sample(reader.read_sample(40124, read_ids=True),
                 read_ids=True,
                 read_timestamps=False,
                 task_name=task_name)
    check_sample(reader.read_sample(10017, read_ids=True),
                 read_ids=True,
                 read_timestamps=False,
                 task_name=task_name)
    tests_io(
        f"Suceeded testing read sample for task {task_name} with ids and without timestamps passed")

    # Testing with reading ids and timestamps
    check_sample(reader.read_sample(10017, read_ids=True, read_timestamps=True),
                 read_ids=True,
                 read_timestamps=True,
                 task_name=task_name)
    check_sample(reader.read_sample(10017, read_ids=True, read_timestamps=True),
                 read_ids=True,
                 read_timestamps=True,
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} with ids and timestamps passed")

    # Testing convert to numpy on read
    check_sample(reader.read_sample(40124, data_type=np.ndarray),
                 read_ids=False,
                 read_timestamps=False,
                 data_type=np.ndarray,
                 task_name=task_name)

    check_sample(reader.read_sample(40124, data_type=pd.DataFrame),
                 read_ids=False,
                 read_timestamps=False,
                 data_type=pd.DataFrame,
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} with numpy conversion passed")


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_read_samples(task_name: str, preprocessed_readers: ProcessedSetReader):
    tests_io(f"Test case read samples for task {task_name}", level=0)
    reader = preprocessed_readers[task_name]
    # Testing without reading ids and timestamps
    check_sample(reader.read_samples([10017, 40124]),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    check_sample(reader.read_samples([10017]),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    check_sample(reader.read_samples(reader.subject_ids),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    tests_io(
        f"Suceeded testing read samples for task {task_name} without ids and timestamps passed")

    # Testing with reading ids but without timestamps
    check_samples(reader.read_samples([10017, 40124], read_ids=True),
                  read_timestamps=False,
                  task_name=task_name)
    check_samples(reader.read_samples([10017], read_ids=True),
                  read_timestamps=False,
                  task_name=task_name)
    check_samples(reader.read_samples(reader.subject_ids, read_ids=True),
                  read_timestamps=False,
                  task_name=task_name)
    tests_io(
        f"Suceeded testing read samples for task {task_name} with ids and without timestamps passed"
    )

    # Testing with reading ids and timestamps
    check_samples(reader.read_samples([10017, 40124], read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  task_name=task_name)
    check_samples(reader.read_samples([10017], read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  task_name=task_name)
    check_samples(reader.read_samples(reader.subject_ids, read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  task_name=task_name)
    tests_io(f"Suceeded testing read samples for task {task_name} with ids and timestamps passed")

    # Testing convert to numpy on read
    check_sample(reader.read_samples([40124, 10017], data_type=np.ndarray),
                 read_ids=False,
                 read_timestamps=False,
                 data_type=np.ndarray,
                 task_name=task_name)

    check_sample(reader.read_samples([40124, 10017], data_type=pd.DataFrame),
                 read_ids=False,
                 read_timestamps=False,
                 data_type=pd.DataFrame,
                 task_name=task_name)
    tests_io(f"Suceeded testing read samples for task {task_name} with numpy conversion passed")


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_random_sample(task_name: str, preprocessed_readers: ProcessedSetReader):
    tests_io(f"Test case random samples for task {task_name}", level=0)
    reader = preprocessed_readers[task_name]
    # Testing without reading ids and timestamps
    check_sample(reader.random_samples(10),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    check_sample(reader.random_samples(),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    check_sample(reader.random_samples(n_samples=10),
                 read_ids=False,
                 read_timestamps=False,
                 task_name=task_name)
    tests_io(
        f"Suceeded testing random samples for task {task_name} without ids and timestamps passed")

    # Testing with reading ids but without timestamps
    check_samples(reader.random_samples(10, read_ids=True),
                  read_timestamps=False,
                  task_name=task_name)
    check_samples(reader.random_samples(read_ids=True), read_timestamps=False, task_name=task_name)
    check_samples(reader.random_samples(n_samples=10, read_ids=True),
                  read_timestamps=False,
                  task_name=task_name)
    tests_io(
        f"Suceeded testing random samples for task {task_name} with ids and without timestamps passed"
    )

    # Testing with reading ids and timestamps
    check_samples(reader.random_samples(10, read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  task_name=task_name)
    check_samples(reader.random_samples(read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  task_name=task_name)
    check_samples(reader.random_samples(n_samples=10, read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  task_name=task_name)
    tests_io(f"Suceeded testing random samples for task {task_name} with ids and timestamps passed")

    # Test without replacement property
    samples = reader.random_samples(len(reader.subject_ids), read_ids=True)
    check_samples(samples=samples, read_timestamps=False, task_name=task_name)
    assert len(set(samples["X"].keys())) == len(samples["X"].keys())
    assert len(set(samples["X"].keys())) == len(reader.subject_ids)

    # Exceeding the set size results in warning and set sized sample collection
    samples = reader.random_samples(2 * len(reader.subject_ids), read_ids=True)
    check_samples(samples, read_timestamps=False, task_name=task_name)
    assert len(set(samples["X"].keys())) == len(samples["X"].keys())
    assert len(set(samples["X"].keys())) == len(reader.subject_ids)
    tests_io(f"Suceeded testing random samples for task {task_name} without replacement passed")


def check_samples(samples: dict, read_timestamps: bool, data_type=None, task_name=None):
    assert isinstance(samples, dict)
    assert set(samples.keys()) == set(["X", "y", "t"] if read_timestamps else ["X", "y"])
    for subject_id in samples["X"]:
        assert isinstance(subject_id, int)
        sample = {"X": samples["X"][subject_id], "y": samples["y"][subject_id]}
        if read_timestamps:
            sample.update({"t": samples["t"][subject_id]})
        check_sample(sample,
                     read_ids=True,
                     read_timestamps=read_timestamps,
                     data_type=data_type,
                     task_name=task_name)


def check_sample(sample: dict,
                 read_ids: bool,
                 read_timestamps: bool,
                 data_type=None,
                 task_name=None):
    assert set(sample.keys()) == set(["X", "y", "t"] if read_timestamps else ["X", "y"])
    X, Y = sample["X"], sample["y"]

    if read_ids:
        assert isinstance(X, dict)
        assert isinstance(Y, dict)
        assert len(X) == len(Y)
        assert all([
            isinstance(stay_id, int) and
            (isinstance(timeseries, np.ndarray) if data_type == np.ndarray else isinstance(
                timeseries, pd.DataFrame)) for stay_id, timeseries in X.items()
        ])
    else:
        assert isinstance(X, list)
        assert isinstance(Y, list)
        assert len(X) == len(Y)
        assert all([(isinstance(timeseries, np.ndarray) if data_type == np.ndarray else isinstance(
            timeseries, pd.DataFrame)) for timeseries in X])
    if data_type == pd.DataFrame or data_type is None:
        for X_sample, Y_sample in (zip(X.values(), Y.values()) if read_ids else zip(X, Y)):
            X_sample = upper_case_column_names(deepcopy(X_sample))

            assert set([
                column.upper() for column in DATASET_SETTINGS["timeseries"]["dtype"].keys()
            ]) == set(X_sample)
            assert set([column.upper() for column in timeseries_label_cols[task_name]
                       ]) == set(Y_sample.columns.str.upper())

            assert all([X_sample[column].dtype == DTYPES[column] for column in X_sample.columns])
    else:
        for X_sample, Y_sample in (zip(X.values(), Y.values()) if read_ids else zip(X, Y)):
            assert isinstance(X_sample, np.ndarray)
            assert isinstance(Y_sample, np.ndarray)
            assert X_sample.shape[1] == len(DATASET_SETTINGS["timeseries"]["dtype"])
            assert Y_sample.shape[1] == len(timeseries_label_cols[task_name])


if __name__ == "__main__":
    import shutil
    # if SEMITEMP_DIR.is_dir():
    #     shutil.rmtree(str(SEMITEMP_DIR))
    reader_dict = dict()
    for task in TASK_NAMES:
        reader = datasets.load_data(chunksize=75835,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    preprocess=True,
                                    task=task)
        reader_dict[task] = reader
        test_read_sample(task, reader_dict)
        test_read_samples(task, reader_dict)
        test_random_sample(task, reader_dict)

    print("All tests passed!")
    # if TEMP_DIR.is_dir():
    #     shutil.rmtree(str(TEMP_DIR))
