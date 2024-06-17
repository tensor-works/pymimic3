import pytest
import datasets
import shutil
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
from datasets.readers import ProcessedSetReader
from tests.pytest_utils.general import assert_file_creation
from tests.pytest_utils.discretization import assert_strategy_equals, prepare_processed_data
from datasets.processors.discretizers import MIMICDiscretizer
from utils.IO import *
from tests.pytest_utils import copy_dataset
from tests.tsettings import *


@pytest.mark.parametrize("start_strategy", ["zero", "relative"])
@pytest.mark.parametrize("impute_strategy", ["next", "normal_value", "previous", "zero"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_iterative_discretizer(task_name: str, start_strategy: str, impute_strategy: str,
                               discretizer_listfiles: Dict[str, pd.DataFrame],
                               preprocessed_readers: Dict[str, ProcessedSetReader]):

    tests_io(
        f"Testing discretizer for task {task_name}\n"
        f"Impute strategy '{impute_strategy}' \n"
        f"Start strategy '{start_strategy}'",
        level=0)
    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))
    listfile = discretizer_listfiles[task_name]
    test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
    test_dir_path = Path(test_data_dir, f"imp{impute_strategy}_start{start_strategy}")

    if impute_strategy == "normal_value":
        impute_strategy = "normal"

    prepare_processed_data(task_name, listfile, preprocessed_readers[task_name])
    tests_io(f"Testing with - task: {task_name}, start_strategy: {start_strategy},"
             f" impute_strategy: {impute_strategy}, deep_supervision: False")
    reader = datasets.load_data(chunksize=75837,
                                source_path=TEST_DATA_DEMO,
                                storage_path=TEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=(start_strategy == "zero"),
                                impute_strategy=impute_strategy,
                                task=task_name)

    assert_file_creation(reader.root_path, test_dir_path, file_suffix="h5")
    X_discretized, _ = reader.read_samples(read_ids=True).values()
    assert_strategy_equals(X_discretized, test_dir_path)

    if task_name in ["DECOMP", "LOS"]:
        tests_io(f"Testing with - task: {task_name}, start_strategy: {start_strategy},"
                 f" impute_strategy: {impute_strategy}, deep_supervision: True")
        reader = datasets.load_data(chunksize=75837,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=TEMP_DIR,
                                    discretize=True,
                                    time_step_size=1.0,
                                    start_at_zero=(start_strategy == "zero"),
                                    impute_strategy=impute_strategy,
                                    task=task_name,
                                    deep_supervision=True)

        assert_file_creation(reader.root_path,
                             test_dir_path,
                             file_suffix="h5",
                             file_prefixes=["X", "M", "yds"])
        X_discretized, _ = reader.read_samples(read_ids=True).values()
        assert_strategy_equals(X_discretized, test_dir_path)

    tests_io("Succeeded in testing!")


@pytest.mark.parametrize("start_strategy", ["zero", "relative"])
@pytest.mark.parametrize("impute_strategy", ["next", "normal_value", "previous", "zero"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_compact_discretizer(task_name: str, start_strategy: str, impute_strategy: str,
                             discretizer_listfiles: Dict[str, pd.DataFrame],
                             preprocessed_readers: Dict[str, ProcessedSetReader]):

    tests_io(
        f"Testing discretizer for task {task_name}\n"
        f"Impute strategy '{impute_strategy}' \n"
        f"Start strategy '{start_strategy}'",
        level=0)

    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))
    listfile = discretizer_listfiles[task_name]
    test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
    test_dir_path = Path(test_data_dir, f"imp{impute_strategy}_start{start_strategy}")
    discretized_path = Path(TEMP_DIR, "discretized", task_name)

    if impute_strategy == "normal_value":
        impute_strategy = "normal"

    prepare_processed_data(task_name, listfile, preprocessed_readers[task_name])

    tests_io(f"Testing with - task: {task_name}, start_strategy: {start_strategy},"
             f" impute_strategy: {impute_strategy}, deep_supervision: False")
    X_discretized, _ = datasets.load_data(source_path=TEST_DATA_DEMO,
                                          storage_path=TEMP_DIR,
                                          discretize=True,
                                          time_step_size=1.0,
                                          start_at_zero=(start_strategy == "zero"),
                                          impute_strategy=impute_strategy,
                                          task=task_name).values()
    assert_file_creation(discretized_path, test_dir_path, file_suffix="h5")
    assert_strategy_equals(X_discretized, test_dir_path)

    if task_name in ["DECOMP", "LOS"]:
        tests_io(f"Testing with - task: {task_name}, start_strategy: {start_strategy},"
                 f" impute_strategy: {impute_strategy}, deep_supervision: True")
        X_discretized, _ = datasets.load_data(source_path=TEST_DATA_DEMO,
                                              storage_path=TEMP_DIR,
                                              discretize=True,
                                              time_step_size=1.0,
                                              start_at_zero=(start_strategy == "zero"),
                                              impute_strategy=impute_strategy,
                                              task=task_name,
                                              deep_supervision=True).values()
        assert_file_creation(discretized_path,
                             test_dir_path,
                             file_suffix="h5",
                             file_prefixes=["X", "M", "yds"])
        assert_strategy_equals(X_discretized, test_dir_path)

    tests_io("Succeeded in testing!")


def test_discretizer_state_persistence():
    tests_io("Testing discretizer state persistence with deep supervision", level=0)
    task_name = "DECOMP"
    storage_path = TEMP_DIR

    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))

    proc_reader = ProcessedSetReader(root_path=Path(SEMITEMP_DIR, "processed", "IHM"))
    tests_io("Running discretizer without deep supervision")
    reader = datasets.load_data(chunksize=75837,
                                source_path=TEST_DATA_DEMO,
                                storage_path=TEMP_DIR,
                                discretize=True,
                                task=task_name)

    tests_io("Rerunning discretizer without deep supervision")
    discretized_storage_path = Path(storage_path, "discretized", task_name)
    discretizer = MIMICDiscretizer(task=task_name, storage_path=discretized_storage_path)
    assert discretizer._tracker.is_finished == True
    reader = discretizer.transform_reader(reader=proc_reader)

    tests_io("Testing discretizer state persistence with deep supervision")
    tests_io("Rerunning discretizer with deep supervision")
    discretizer = MIMICDiscretizer(task=task_name,
                                   storage_path=discretized_storage_path,
                                   deep_supervision=True)
    assert discretizer._tracker.is_finished == False
    reader = discretizer.transform_reader(reader=proc_reader)
    tests_io("Rerunning discretizer without deep supervision")
    discretizer = MIMICDiscretizer(task=task_name,
                                   storage_path=discretized_storage_path,
                                   deep_supervision=True)
    assert discretizer._tracker.is_finished == True
    reader = discretizer.transform_reader(reader=proc_reader)
    tests_io("Succeeded in testing state persistent")


if __name__ == "__main__":

    import re
    listfiles = dict()
    # Preparing the listfiles
    for task_name in TASK_NAMES:
        # Path to discretizer sets
        test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
        # Listfile with truth values
        idx_name = "filename" if task_name == "MULTI" else "stay"
        listfile = pd.read_csv(Path(test_data_dir, "listfile.csv")).set_index(idx_name)
        stay_name_regex = r"(\d+)_episode(\d+)_timeseries\.csv"

        listfile = listfile.reset_index()
        listfile["subject"] = listfile[idx_name].apply(
            lambda x: re.search(stay_name_regex, x).group(1))
        listfile["icustay"] = listfile[idx_name].apply(
            lambda x: re.search(stay_name_regex, x).group(2))
        listfile = listfile.set_index(idx_name)
        listfiles[task_name] = listfile
    readers = dict()
    discretizer_dir = Path(TEMP_DIR, "discretized")

    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    for task_name in set(TASK_NAMES) - set(["MULTI"]):
        # Simulate semi_temp fixture

        if Path(SEMITEMP_DIR, "processed", task_name).is_dir():
            copy_dataset("extracted")
            copy_dataset(Path("processed", task_name))
            reader = ProcessedSetReader(root_path=Path(TEMP_DIR, "processed", task_name))
        else:
            reader = datasets.load_data(chunksize=75835,
                                        source_path=TEST_DATA_DEMO,
                                        storage_path=SEMITEMP_DIR,
                                        preprocess=True,
                                        task=task_name)
            copy_dataset("extracted")
            copy_dataset(Path("processed", task_name))
        if discretizer_dir.is_dir():
            shutil.rmtree(str(discretizer_dir))

        if task_name == "IHM":
            test_discretizer_state_persistence()

        readers[task_name] = reader
        for start_strategy in ["zero", "relative"]:
            for impute_strategy in ["next", "normal_value", "previous", "zero"]:
                if discretizer_dir.is_dir():
                    shutil.rmtree(str(discretizer_dir))
                test_iterative_discretizer(task_name=task_name,
                                           impute_strategy=impute_strategy,
                                           start_strategy=start_strategy,
                                           discretizer_listfiles=listfiles,
                                           preprocessed_readers=readers)

                if discretizer_dir.is_dir():
                    shutil.rmtree(str(discretizer_dir))
                test_compact_discretizer(task_name=task_name,
                                         impute_strategy=impute_strategy,
                                         start_strategy=start_strategy,
                                         discretizer_listfiles=listfiles,
                                         preprocessed_readers=readers)
