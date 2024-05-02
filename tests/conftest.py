import shutil
import pytest
import os
import datasets
from typing import Dict
from tests.settings import *
from utils.IO import *
from datasets.readers import ExtractedSetReader, EventReader, ProcessedSetReader
from settings import *

collect_ignore = ['src/utils/IO.py']


def pytest_configure(config) -> None:
    os.environ["DEBUG"] = "0"

    if SEMITEMP_DIR.is_dir():
        shutil.rmtree(str(SEMITEMP_DIR))


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
                                 task=TASK_NAME_MAPPING[name]) for name in TASK_NAMES
    }


def pytest_unconfigure(config) -> None:
    os.environ["DEBUG"] = "0"
    if SEMITEMP_DIR.is_dir():
        shutil.rmtree(str(SEMITEMP_DIR))
