import pytest
import datasets
import shutil
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
from datasets.readers import ProcessedSetReader
from datasets.writers import DataSetWriter
from tests.utils.discretization import assert_strategy_equals, prepare_processed_data
from utils.IO import *
from tests.utils import copy_dataset
from tests.settings import *
from tests.conftest import discretizer_listfiles


@pytest.mark.parametrize("start_strategy", ["zero", "relative"])
@pytest.mark.parametrize("impute_strategy", ["next", "normal_value", "previous", "zero"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_discretizer(task_name: str, start_strategy: str, impute_strategy: str,
                     discretizer_listfiles: Dict[str, pd.DataFrame],
                     preprocessed_readers: Dict[str, ProcessedSetReader]):

    copy_dataset("extracted")
    listfile = discretizer_listfiles[task_name]
    test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
    test_dir_path = Path(test_data_dir, f"imp{impute_strategy}_start{start_strategy}")

    tests_io(f"Testing discretizer for task {task_name}\n"
             f"Impute strategy '{impute_strategy}' \n"
             f"Start strategy '{start_strategy}'")
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
                                task=task_name)

    X_discretized, _ = reader.read_samples(read_ids=True).values()

    assert_strategy_equals(X_discretized, test_dir_path)

    tests_io("Succeeded in testing!")


if __name__ == "__main__":

    import re
    listfiles = dict()
    # Preparing the listfiles
    for task_name in TASK_NAMES:
        # Path to discretizer sets
        test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
        # Listfile with truth values
        listfile = pd.read_csv(Path(test_data_dir, "listfile.csv")).set_index("stay")
        stay_name_regex = r"(\d+)_episode(\d+)_timeseries\.csv"

        listfile = listfile.reset_index()
        listfile["subject"] = listfile["stay"].apply(
            lambda x: re.search(stay_name_regex, x).group(1))
        listfile["icustay"] = listfile["stay"].apply(
            lambda x: re.search(stay_name_regex, x).group(2))
        listfile = listfile.set_index("stay")
        listfiles[task_name] = listfile
    readers = dict()
    for task_name in TASK_NAMES:
        # Simulate semi_temp fixture
        reader = datasets.load_data(chunksize=75837,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    reprocess=True,
                                    task=task_name)
        readers[task_name] = reader
        for start_strategy in ["zero", "relative"]:
            for impute_strategy in ["next", "normal_value", "previous", "zero"]:
                if TEMP_DIR.is_dir():
                    shutil.rmtree(str(TEMP_DIR))
                test_discretizer(task_name=task_name,
                                 impute_strategy=impute_strategy,
                                 start_strategy=start_strategy,
                                 discretizer_listfiles=listfiles,
                                 preprocessed_readers=readers)
