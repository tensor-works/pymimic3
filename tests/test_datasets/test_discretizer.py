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
        f"Testing iteravtive discretizer for task {task_name}\n"
        f"impute_strategy '{impute_strategy}' \n"
        f"start_strategy '{start_strategy}' "
        f"deep_supervision: False",
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
    tests_io("Succeeded in testing discretizer!")


@pytest.mark.parametrize("start_strategy", ["zero", "relative"])
@pytest.mark.parametrize("impute_strategy", ["next", "normal_value", "previous", "zero"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_iterative_discretizer_with_ds(task_name: str, start_strategy: str, impute_strategy: str,
                                       discretizer_listfiles: Dict[str, pd.DataFrame],
                                       preprocessed_readers: Dict[str, ProcessedSetReader]):

    tests_io(
        f"Testing iterative discretizer for task {task_name}\n"
        f"impute_strategy '{impute_strategy}' \n"
        f"start_strategy '{start_strategy}' "
        f"deep_supervision: True",
        level=0)
    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))
    listfile = discretizer_listfiles[task_name]
    test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
    test_dir_path = Path(test_data_dir, f"imp{impute_strategy}_start{start_strategy}")

    if impute_strategy == "normal_value":
        impute_strategy = "normal"

    prepare_processed_data(task_name, listfile, preprocessed_readers[task_name])

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
        f"Testing compact discretizer for task {task_name}\n"
        f"Impute strategy '{impute_strategy}' \n"
        f"Start strategy '{start_strategy}' "
        f"deep_supervision: False",
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
    X_discretized, _ = datasets.load_data(source_path=TEST_DATA_DEMO,
                                          storage_path=TEMP_DIR,
                                          discretize=True,
                                          time_step_size=1.0,
                                          start_at_zero=(start_strategy == "zero"),
                                          impute_strategy=impute_strategy,
                                          task=task_name).values()
    assert_file_creation(discretized_path, test_dir_path, file_suffix="h5")
    assert_strategy_equals(X_discretized, test_dir_path)
    tests_io("Succeeded in testing discretizer!")


@pytest.mark.parametrize("start_strategy", ["zero", "relative"])
@pytest.mark.parametrize("impute_strategy", ["next", "normal_value", "previous", "zero"])
@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_compact_discretizer_with_ds(task_name: str, start_strategy: str, impute_strategy: str,
                                     discretizer_listfiles: Dict[str, pd.DataFrame],
                                     preprocessed_readers: Dict[str, ProcessedSetReader]):

    tests_io(
        f"Testing compact discretizer for task {task_name}\n"
        f"Impute strategy '{impute_strategy}' \n"
        f"Start strategy '{start_strategy}' "
        f"deep_supervision: True",
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
    discretizer = MIMICDiscretizer(task=task_name,
                                   storage_path=discretized_storage_path,
                                   verbose=True)
    assert discretizer._tracker.is_finished == True
    reader = discretizer.transform_reader(reader=proc_reader)

    tests_io("Testing discretizer state persistence with deep supervision")
    tests_io("Rerunning discretizer with deep supervision")
    discretizer = MIMICDiscretizer(task=task_name,
                                   storage_path=discretized_storage_path,
                                   deep_supervision=True,
                                   verbose=True)
    assert discretizer._tracker.is_finished == False
    reader = discretizer.transform_reader(reader=proc_reader)
    tests_io("Rerunning discretizer with deep supervision to check state persistence")
    discretizer = MIMICDiscretizer(task=task_name,
                                   storage_path=discretized_storage_path,
                                   deep_supervision=True,
                                   verbose=True)
    assert discretizer._tracker.is_finished == True
    reader = discretizer.transform_reader(reader=proc_reader)
    tests_io("Succeeded in testing state persistent")


if __name__ == "__main__":
    from tests.pytest_utils.discretization import prepare_discretizer_listfiles
    listfiles = prepare_discretizer_listfiles(list(set(TASK_NAMES) - set(["MULTI"])))
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

                if task_name in ["DECOMP", "LOS"]:
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
                if task_name in ["DECOMP", "LOS"]:
                    test_compact_discretizer_with_ds(task_name=task_name,
                                                     impute_strategy=impute_strategy,
                                                     start_strategy=start_strategy,
                                                     discretizer_listfiles=listfiles,
                                                     preprocessed_readers=readers)
