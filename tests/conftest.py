import shutil
import pytest
import os
import re
import datasets
import pandas as pd
from typing import Dict
from pathlib import Path
from tests.settings import *
from utils.IO import *
from datasets.readers import ExtractedSetReader, EventReader, ProcessedSetReader
from settings import *

collect_ignore = ['src/utils/IO.py']


def pytest_configure(config) -> None:
    os.environ["DEBUG"] = "0"

    if SEMITEMP_DIR.is_dir():
        shutil.rmtree(str(SEMITEMP_DIR))

    [
        datasets.load_data(chunksize=75835,
                           source_path=TEST_DATA_DEMO,
                           storage_path=SEMITEMP_DIR,
                           preprocess=True,
                           task=name) for name in TASK_NAMES
    ]


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))

    # Execution
    yield

    # Clean after execution
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))


@pytest.fixture(scope="session")
def extracted_reader() -> ExtractedSetReader:
    reader = datasets.load_data(chunksize=75835,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR)
    return reader


@pytest.fixture(scope="session")
def subject_ids(extracted_reader: ExtractedSetReader) -> list:
    icu_history = extracted_reader.read_csv("icu_history.csv")
    subjects = icu_history["SUBJECT_ID"].astype(int).unique().tolist()
    return subjects


@pytest.fixture(scope="session")
def preprocessed_readers() -> Dict[str, ProcessedSetReader]:
    return {
        name: datasets.load_data(chunksize=75835,
                                 source_path=TEST_DATA_DEMO,
                                 storage_path=SEMITEMP_DIR,
                                 preprocess=True,
                                 task=name) for name in TASK_NAMES
    }


@pytest.fixture(scope="session")
def discretizer_listfiles() -> None:
    list_files = dict()
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
        list_files[task_name] = listfile
    return list_files


def pytest_unconfigure(config) -> None:
    os.environ["DEBUG"] = "0"
    if SEMITEMP_DIR.is_dir():
        shutil.rmtree(str(SEMITEMP_DIR))
