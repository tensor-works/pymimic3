import shutil
import pytest
import os
import re
import datasets
import pandas as pd
import ray
from typing import Dict
from pathlib import Path
from tests.tsettings import *
from utils.IO import *
from datasets.readers import ExtractedSetReader, EventReader, ProcessedSetReader
from settings import *

collect_ignore = ['src/utils/IO.py']


def pytest_configure(config) -> None:
    os.environ["DEBUG"] = "0"

    if SEMITEMP_DIR.is_dir():
        shutil.rmtree(str(SEMITEMP_DIR))

    for task_name in TASK_NAMES:
        tests_io(f"Loading preproceesing data for task {task_name}...", end="\r")
        datasets.load_data(chunksize=75835,
                           source_path=TEST_DATA_DEMO,
                           storage_path=SEMITEMP_DIR,
                           preprocess=True,
                           verbose=False,
                           task=task_name)
        tests_io(f"Done loading preprocessing data for task {task_name}")


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    if ray.is_initialized():
        ray.shutdown()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))

    # Execution
    yield

    # Clean after execution
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="session")
def extracted_reader() -> ExtractedSetReader:
    extract_reader = datasets.load_data(chunksize=75835,
                                        source_path=TEST_DATA_DEMO,
                                        storage_path=SEMITEMP_DIR,
                                        verbose=False)
    return extract_reader


@pytest.fixture(scope="session")
def subject_ids(extracted_reader: ExtractedSetReader) -> list:
    icu_history = extracted_reader.read_csv("icu_history.csv")
    subjects = icu_history["SUBJECT_ID"].astype(int).unique().tolist()
    return subjects


@pytest.fixture(scope="session")
def preprocessed_readers() -> Dict[str, ProcessedSetReader]:
    proc_reader = dict()
    for task_name in TASK_NAMES:
        proc_reader[task_name] = datasets.load_data(chunksize=75835,
                                                    source_path=TEST_DATA_DEMO,
                                                    storage_path=SEMITEMP_DIR,
                                                    preprocess=True,
                                                    task=task_name,
                                                    verbose=False)
    return proc_reader


@pytest.fixture(scope="session")
def engineered_readers() -> Dict[str, ProcessedSetReader]:
    eng_reader = dict()
    for task_name in set(TASK_NAMES) - {"MULTI"}:
        tests_io(f"Loading engineered reader for task {task_name}...", end="\r")
        eng_reader[task_name] = datasets.load_data(chunksize=75835,
                                                   source_path=TEST_DATA_DEMO,
                                                   storage_path=SEMITEMP_DIR,
                                                   engineer=True,
                                                   task=task_name,
                                                   verbose=True)
        tests_io(f"Done loading engineered reader for task {task_name}")
    return eng_reader


@pytest.fixture(scope="session")
def discretized_readers() -> Dict[str, ProcessedSetReader]:
    disc_reader = dict()
    for task_name in set(TASK_NAMES) - {"MULTI"}:
        tests_io(f"Loading discretized reader for task {task_name}...", end="\r")
        disc_reader[task_name] = datasets.load_data(chunksize=75835,
                                                    source_path=TEST_DATA_DEMO,
                                                    storage_path=SEMITEMP_DIR,
                                                    discretize=True,
                                                    task=task_name,
                                                    verbose=False)
        tests_io(f"Done loading discretized reader for task {task_name}")

    for task_name in ["DECOMP", "LOS"]:
        tests_io(f"Loading deep supervision discretized reader for task {task_name}...", end="\r")
        _ = datasets.load_data(chunksize=75835,
                               source_path=TEST_DATA_DEMO,
                               storage_path=SEMITEMP_DIR,
                               deep_supervision=True,
                               discretize=True,
                               task=task_name,
                               verbose=False)
        tests_io(f"Done loading deep supervision discretized reader for task {task_name}")
    return disc_reader


@pytest.fixture(scope="session")
def discretizer_listfiles() -> None:
    list_files = dict()
    for task_name in TASK_NAMES:
        # Path to discretizer sets
        test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
        # Listfile with truth values
        listfile = pd.read_csv(Path(test_data_dir, "listfile.csv"),
                               na_values=[''],
                               keep_default_na=False).set_index("stay")
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
