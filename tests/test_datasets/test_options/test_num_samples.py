import datasets
import os
import pandas as pd
import json
import shutil
import pytest
from itertools import chain
from typing import Dict
from pathlib import Path
from utils.IO import *
from tests.settings import *
from tests.utils.general import assert_dataframe_equals
from tests.utils.preprocessing import assert_reader_equals
from tests.utils.feature_engineering import extract_test_ids, concatenate_dataset

from datasets.readers import ExtractedSetReader, ProcessedSetReader
'''

@pytest.mark.xfail(reason="Will fail, num_samples computation for"
                   " iterative generation is off but costly to fix")
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_num_samples_empty_dir(task_name: str):
    # Small sample sizes
    tests_io(f"Test case num samples on empty directory  for {task_name}", level=0)
    for sample_size in chain(range(1, 5, 15), range(1000, 3000, 1000)):
        reader: ExtractedSetReader = datasets.load_data(chunksize=500,
                                                        source_path=TEST_DATA_DEMO,
                                                        num_samples=sample_size,
                                                        storage_path=TEMP_DIR,
                                                        task=task_name)

        assert len(pd.concat(reader.read_subjects(file_types="timeseries")[0])) == sample_size
        tests_io(f"Successfully tested {sample_size}")
        if TEMP_DIR.is_dir():
            shutil.rmtree(TEMP_DIR)
    # TODO! Fix this you should be able to
    # Can't decrease the number of samples
    with pytest.raises(ValueError) as error:
        reader: ExtractedSetReader = datasets.load_data(chunksize=500,
                                                        source_path=TEST_DATA_DEMO,
                                                        num_samples=1,
                                                        storage_path=TEMP_DIR,
                                                        task=task_name)

        assert error.value == f"Number of samples parameter has been lowered from 10 to 9, while the parameter can only be increased!"


@pytest.mark.xfail(reason="Will fail, num_samples computation for"
                   " iterative generation is off but costly to fix")
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_num_samples_existing_dir(task_name: str, sample_size: int):
    tests_io(f"Test case num samples on existing directory  for {task_name}", level=0)
    # Small sample sizes
    for sample_size in chain(range(1, 5, 15), range(1000, 3000, 1000)):
        reader: ExtractedSetReader = datasets.load_data(chunksize=500,
                                                        source_path=TEST_DATA_DEMO,
                                                        num_samples=sample_size,
                                                        storage_path=TEMP_DIR,
                                                        task=task_name)

        assert len(pd.concat(reader.read_subjects(file_types="timeseries")[0])) == sample_size
        tests_io(f"Successfully tested {sample_size}")

    # Can't decrease the number of samples
    with pytest.raises(ValueError) as error:
        reader: ExtractedSetReader = datasets.load_data(chunksize=500,
                                                        source_path=TEST_DATA_DEMO,
                                                        num_samples=1,
                                                        storage_path=TEMP_DIR,
                                                        task=task_name)

        assert error.value == f"Number of samples parameter has been lowered from 10 to 9, while the parameter can only be increased!"
'''
