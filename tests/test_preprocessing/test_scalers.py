# Test the preprocessing.imputer class using the preprocessing_readers from conftest.py. You can find the imputer use case in preprocessing.discretizer and preprocessing.normalizer
import pytest
import numpy as np
import shutil
from typing import Dict
from pathlib import Path
from datasets.readers import ProcessedSetReader
from preprocessing.scalers import MinMaxScaler
from preprocessing.imputers import PartialImputer
from tests.tsettings import *
from tests.pytest_utils import copy_dataset
from utils.IO import *


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_partial_fit_transform(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    tests_io("Test case partial fit transform for scalers", level=0)
    reader = engineered_readers[task_name]
    X = reader.read_samples()["X"]

    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))
    imputer.fit_reader(reader)
    scaler = MinMaxScaler()

    # Test partial_fit
    [scaler.partial_fit(imputer.transform(frame)) for frame in X]
    assert scaler.data_min_ is not None, "Partial fit did not compute min values."
    assert scaler.data_max_ is not None, "Partial fit did not compute max values."
    tests_io(f"Succeeded in test partial_fit")

    # Test transform
    transformed_data = [scaler.transform(imputer.transform(frame)) for frame in X]
    precision = 1e-6
    for data in transformed_data:
        assert (data >= 0 - precision).all() and (data <= 1 + precision).all(), \
            "Data is not within the range [0, 1] after transform."
    tests_io(f"Succeeded in testing partial transform")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_fit_transform_dataset(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    tests_io("Test case for fit transform dataset", level=0)
    # Used to compare metadata
    gt_reader = engineered_readers[task_name]
    gt_dataset = gt_reader.read_samples()
    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))

    copy_dataset(Path("engineered", task_name))
    reader = ProcessedSetReader(Path(TEMP_DIR, "engineered", task_name))
    # Prepare the data by imputing
    reader = imputer.fit_transform_reader(reader)
    dataset = reader.read_samples()

    # ----- test the scaler -----
    scaler = MinMaxScaler(storage_path=Path(TEMP_DIR, "scaler", "0"))

    # Test fit_dataset
    scaler.fit_dataset(dataset)
    assert scaler.data_min_ is not None, "Partial fit did not compute min values."
    assert scaler.data_max_ is not None, "Partial fit did not compute max values."
    tests_io(f"Succeeded in testing fit_dataset")

    # Test transform_dataset
    transformed_dataset = scaler.transform_dataset(dataset)
    precision = 1e-6
    for data in transformed_dataset["X"]:
        assert (data >= 0 - precision).all() and (data <= 1 + precision).all(), \
            "Data is not within the range [0, 1] after transform_dataset."
    tests_io(f"Succeeded in testing transform_dataset")

    # Test fit_transform_dataset
    scaler = MinMaxScaler(storage_path=Path(TEMP_DIR, "scaler", "1"))
    transformed_dataset = scaler.fit_transform_dataset(dataset)
    for data in transformed_dataset["X"]:
        assert (data >= 0 - precision).all() and (data <= 1 + precision).all(), \
            "Data is not within the range [0, 1] after fit_transform_dataset."

    assert len(transformed_dataset["X"]) == len(gt_dataset["X"]), \
        (f"Data length mismatch after fit_transform_dataset. Length transformed: {len(transformed_dataset['X'])}, "
        f"Length ground truth: {len(gt_dataset['X'])}")
    tests_io(f"Succeeded in testing fit_transform_dataset")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_fit_and_transform_reader(task_name: str, engineered_readers: Dict[str,
                                                                           ProcessedSetReader]):
    tests_io("Test case fitting and transforming the reader seperately", level=0)

    # Used to compare metadata
    gt_reader = engineered_readers[task_name]
    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))

    copy_dataset(Path("engineered", task_name))
    reader = ProcessedSetReader(Path(TEMP_DIR, "engineered", task_name))
    # Prepare the data by imputing
    reader = imputer.fit_transform_reader(reader)

    # ------ Test the scaler -------
    scaler = MinMaxScaler(storage_path=Path(TEMP_DIR, "scaler", "0"))
    # Test fit_transform_reader
    scaler.fit_reader(reader)
    assert scaler.data_min_ is not None, "Partial fit did not compute min values."
    assert scaler.data_max_ is not None, "Partial fit did not compute max values."

    transformed_reader = scaler.transform_reader(reader)
    check_reader(transformed_reader, gt_reader)
    tests_io(f"Succeeded in testing fit and transform reader")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_fit_transform_reader(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    # Test fit and transform at once
    tests_io("Test case fit_transform_reader for imputer", level=0)

    # Used to compare metadata
    gt_reader = engineered_readers[task_name]
    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))

    copy_dataset(Path("engineered", task_name))
    reader = ProcessedSetReader(Path(TEMP_DIR, "engineered", task_name))
    # Prepare the data by imputing
    reader = imputer.fit_transform_reader(reader)

    # ------ Test the scaler -------
    scaler = MinMaxScaler(storage_path=Path(TEMP_DIR, "scaler", "0"))
    scaler.fit_transform_reader(reader)

    assert scaler.data_min_ is not None, "Scaler state was not loaded."
    assert scaler.data_max_ is not None, "Scaler state was not loaded."
    check_reader(reader, gt_reader)
    tests_io(f"Succeeded in testing fit_transform_reader")


@pytest.mark.parametrize("task_name", ["DECOMP"])
def test_save_load(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    reader = engineered_readers[task_name]

    imputer = PartialImputer(strategy='mean', storage_path=Path(TEMP_DIR, "imputer", "0"))
    imputer.fit_reader(reader)

    storage_path_0 = Path(TEMP_DIR, "scaler", "0")
    scaler = MinMaxScaler(storage_path=storage_path_0)

    imputed_data = [imputer.transform(frame) for frame in reader.read_samples()["X"]]
    for frame in imputed_data:
        scaler.partial_fit(frame)

    # Test save
    scaler.save()
    assert Path(storage_path_0, "minmax_scaler.pkl").exists(), "Scaler state was not saved."
    tests_io(f"Succeeded in testing save")

    # Test load
    scaler_load_test = MinMaxScaler(storage_path=storage_path_0)
    scaler_load_test.load()
    assert scaler_load_test.data_min_ is not None, "Scaler state was not loaded."
    assert scaler_load_test.data_max_ is not None, "Scaler state was not loaded."
    tests_io(f"Succeeded in testing load with specified storage path")

    storage_path_1 = Path(TEMP_DIR, "scaler", "1")
    scaler.save(storage_path_1)
    assert Path(storage_path_1, "minmax_scaler.pkl").exists(), "Scaler state was not saved."

    scaler_load_test = MinMaxScaler(storage_path=storage_path_1)
    scaler_load_test.load()
    assert scaler_load_test.data_min_ is not None, "Scaler state was not loaded."
    assert scaler_load_test.data_max_ is not None, "Scaler state was not loaded."
    tests_io(f"Succeeded in testing save and load with specified storage path")


def check_reader(reader: ProcessedSetReader, ground_truth_reader: ProcessedSetReader):
    precision = 1e-6
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
            assert (frame.values >= 0 - precision).all() and (frame.values <= 1 + precision).all(), \
                "Data is not within the range [0, 1] after transform_reader."
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
    if Path(TEMP_DIR, "scaler").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "scaler")))
    test_partial_fit_transform("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "scaler").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "scaler")))
    test_fit_transform_dataset("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "scaler").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "scaler")))
    test_fit_and_transform_reader("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "scaler").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "scaler")))
    # test_save_load("DECOMP", {"DECOMP": eng_reader})
    if Path(TEMP_DIR, "scaler").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "scaler")))
    test_fit_transform_reader("DECOMP", {"DECOMP": eng_reader})
