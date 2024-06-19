# Test the preprocessing.imputer class using the preprocessing_readers from conftest.py. You can find the imputer use case in preprocessing.discretizer and preprocessing.normalizer
import pytest
import numpy as np
import shutil
from typing import Dict
from pathlib import Path
from datasets.readers import ProcessedSetReader
from preprocessing.imputers import PartialImputer
from tests.tsettings import *
from tests.pytest_utils import copy_dataset
from utils.IO import *


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_partial_fit_transform(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    tests_io("Test case partial fit transform for imputer", level=0)
    reader = engineered_readers[task_name]
    X = reader.read_samples()["X"]

    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))

    # Test partial_fit
    for frame in X:
        imputer.partial_fit(frame)
    assert imputer.statistics_ is not None, "Partial fit did not compute statistics."
    tests_io(f"Succeeded in test partial_fit")
    # Test transform
    transformed_data = [imputer.transform(frame) for frame in X]
    for data in transformed_data:
        assert not np.isnan(
            data).any(), "There are NaNs left in the transformed data after transform."
    tests_io(f"Succeeded in testing partial transform")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_fit_transform_dataset(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    tests_io("Test case fit transform dataset for imputer", level=0)

    reader = engineered_readers[task_name]
    dataset = reader.read_samples()

    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))

    # Test fit_dataset
    imputer.fit_dataset(dataset)
    tests_io(f"Succeeded in testing fit_dataset")

    # Test transform_dataset
    transformed_dataset = imputer.transform_dataset(dataset)
    for data in transformed_dataset["X"]:
        assert not np.isnan(
            data).any(), "There are NaNs left in the transformed data after transform_dataset."
    tests_io(f"Succeeded in testing transform_dataset")

    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "1"))
    transformed_dataset = imputer.fit_transform_dataset(dataset)
    for data in transformed_dataset["X"]:
        assert not np.isnan(
            data).any(), "There are NaNs left in the transformed data after transform_dataset."
    tests_io(f"Succeeded in testing data for nans")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_fit_and_transform_reader(task_name: str, engineered_readers: Dict[str,
                                                                           ProcessedSetReader]):

    tests_io("Test case fit and transform reader for imputer", level=0)

    # Test fit and transform seperately
    gt_reader = engineered_readers[task_name]
    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))

    copy_dataset(Path("engineered", task_name))
    reader = ProcessedSetReader(Path(TEMP_DIR, "engineered", task_name))
    # Test fit_reader
    imputer.fit_reader(reader, save=True)
    assert imputer.statistics_ is not None, "Fit_reader did not compute statistics."
    tests_io(f"Succeeded in testing fit_reader")

    # Test transform_reader
    reader = imputer.transform_reader(reader)
    check_reader(reader, gt_reader)
    tests_io(f"Succeeded in testing data for nans")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_fit_transform_reader(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    # Test fit and transform at once
    tests_io("Test case fit_transform_reader for imputer", level=0)

    gt_reader = engineered_readers[task_name]
    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))

    copy_dataset(Path("engineered", task_name))
    reader = ProcessedSetReader(Path(TEMP_DIR, "engineered", task_name))

    reader = imputer.fit_transform_reader(reader)
    assert imputer.statistics_ is not None, "Fit_reader did not compute statistics."
    check_reader(reader, gt_reader)
    tests_io(f"Succeeded in testing fit_transform_reader")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_save_load(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    tests_io("Test case save and load imputer", level=0)

    reader = engineered_readers[task_name]

    storage_path_0 = Path(TEMP_DIR, "imputer", "0")
    imputer = PartialImputer(strategy='mean', verbose=0, storage_path=storage_path_0)

    imputer.fit_reader(reader)

    # Test save
    imputer.save()
    assert Path(storage_path_0, "partial_imputer.pkl").exists(), "Imputer state was not saved."
    tests_io(f"Succeeded in testing save")

    # Test load
    imputer_load_test = PartialImputer(strategy='mean', verbose=0, storage_path=storage_path_0)
    imputer_load_test.load()
    assert imputer_load_test.statistics_ is not None, "Imputer state was not loaded."
    tests_io(f"Succeeded in testing load with specified storage path")

    storage_path_1 = Path(TEMP_DIR, "imputer", "1")
    imputer.save(storage_path_1)
    assert Path(storage_path_1, "partial_imputer.pkl").exists(), "Imputer state was not saved."

    imputer_load_test = PartialImputer(strategy='mean', verbose=0, storage_path=storage_path_1)
    imputer_load_test.load()
    assert imputer_load_test.statistics_ is not None, "Imputer state was not loaded."
    tests_io(f"Succeeded in testing save and load with specified storage path")


def check_reader(reader: ProcessedSetReader, ground_truth_reader: ProcessedSetReader):
    assert isinstance(reader, ProcessedSetReader)

    data = reader.read_samples(read_ids=True)["X"]
    assert not set(ground_truth_reader.subject_ids) - set(reader.subject_ids), \
        f"Reader added subjects {*set(ground_truth_reader.subject_ids) - set(reader.subject_ids),}"

    assert not set(reader.subject_ids) - set(ground_truth_reader.subject_ids), \
        f"Reader missing subjects {*set(reader.subject_ids) - set(ground_truth_reader.subject_ids),}"

    for subject_id, stay_data in data.items():
        stay_ids = list()
        gt_stay_data = ground_truth_reader.read_sample(subject_id, read_ids=True)["X"]
        gt_stay_ids = gt_stay_data.keys()
        for stay_id, frame in stay_data.items():
            if stay_id in gt_stay_ids:
                assert gt_stay_data[stay_id].shape == frame.shape
            stay_ids.append(stay_id)
            assert not np.isnan(frame.values).any(), \
                "There are NaNs left in the transformed data after transform_reader."
        assert not set(gt_stay_ids) - set(stay_ids), \
            f"Reader added stays {*set(gt_stay_ids) - set(stay_ids),}"
        assert not set(stay_ids) - set(gt_stay_ids), \
            f"Reader missing stays {*set(stay_ids) - set(gt_stay_ids),}"
    tests_io(f"Succeeded in testing transform_reader")


if __name__ == "__main__":
    import datasets
    eng_reader = datasets.load_data(chunksize=75835,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    engineer=True,
                                    task="DECOMP",
                                    verbose=True)
    if Path(TEMP_DIR, "imputer").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "imputer")))
    test_partial_fit_transform("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "imputer").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "imputer")))
    test_fit_transform_dataset("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "imputer").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "imputer")))
    test_fit_and_transform_reader("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "imputer").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "imputer")))
    test_fit_transform_reader("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "imputer").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "imputer")))
    test_save_load("DECOMP", {"DECOMP": eng_reader})
