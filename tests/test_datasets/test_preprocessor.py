import datasets
import pandas as pd
import re
import pytest
from pathlib import Path
from datasets.readers import ExtractedSetReader
from utils.IO import *
from tests.pytest_utils import copy_dataset
from tests.decorators import repeat
from tests.tsettings import *
from settings import *
from tests.pytest_utils.general import assert_file_creation
from tests.pytest_utils.preprocessing import assert_reader_equals, assert_dataset_equals

kwargs = {
    "IHM": {
        "minimum_length_of_stay": IHM_SETTINGS['label_start_time']
    },
    "DECOMP": {
        "label_start_time": DECOMP_SETTINGS['label_start_time']
    },
    "LOS": {
        "label_start_time": LOS_SETTINGS['label_start_time']
    },
    "PHENO": {},
    "MULTI": {
        "file_suffix": "h5",
        "test_index": "filename"
    }
}


@pytest.mark.parametrize("task_name", TASK_NAMES)
@repeat(2)
def test_iterative_processing_task(task_name: str):
    """_summary_

    Args:
        task_name (str): _description_
    """
    tests_io(f"Test iterative preprocessor for task {task_name}", level=0)

    # Resolve task name
    generated_path = Path(TEMP_DIR, "processed", task_name)  # Outpath for task generation
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    copy_dataset("extracted")

    # Load
    reader = datasets.load_data(chunksize=75835,
                                source_path=TEST_DATA_DEMO,
                                storage_path=TEMP_DIR,
                                preprocess=True,
                                task=generated_path.name)

    assert_file_creation(reader.root_path, test_data_dir, **kwargs[generated_path.name])

    tests_io(f"All {task_name} files have been created as expected")
    # Compare the dataframes in the directory
    assert_reader_equals(reader, test_data_dir, **kwargs[generated_path.name])
    tests_io(f"{task_name} preprocessing successfully tested against original code!")

    return


@pytest.mark.parametrize("task_name", TASK_NAMES)
@repeat(2)
def test_compact_processing_task(task_name: str):
    """_summary_

    Args:
        task_name (str): _description_
    """
    tests_io(f"Test compact preprocessor for task {task_name}", level=0)

    # Resolve task name
    generated_path = Path(TEMP_DIR, "processed", task_name)  # Outpath for task generation
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    copy_dataset("extracted")

    # Load/Create data
    dataset = datasets.load_data(source_path=TEST_DATA_DEMO,
                                 storage_path=TEMP_DIR,
                                 preprocess=True,
                                 task=generated_path.name)

    assert_file_creation(generated_path, test_data_dir, **kwargs[generated_path.name])

    tests_io(f"All {task_name} files have been created as expected")
    # Compare the dataframes in the directory
    assert_dataset_equals(dataset["X"], dataset["y"], generated_path, test_data_dir,
                          **kwargs[generated_path.name])
    tests_io(f"{task_name} preprocessing successfully tested against original code!")

    return


if __name__ == "__main__":
    import shutil
    _ = datasets.load_data(chunksize=75835, source_path=TEST_DATA_DEMO, storage_path=SEMITEMP_DIR)
    for task in TASK_NAMES:
        if Path(TEMP_DIR).is_dir():
            shutil.rmtree(str(Path(TEMP_DIR)))
        test_compact_processing_task(task)
        if Path(TEMP_DIR).is_dir():
            shutil.rmtree(str(TEMP_DIR))
        test_iterative_processing_task(task)
