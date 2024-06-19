import pytest
import datasets
import re
import shutil

import pandas as pd
import numpy as np
from pathlib import Path
from datasets.readers import ProcessedSetReader
from datasets.writers import DataSetWriter
from datasets.processors.discretizers import MIMICDiscretizer
from tests.pytest_utils.general import assert_dataframe_equals
from utils.IO import *
from tests.tsettings import *


def prepare_discretizer_listfiles(task_name: str):
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
    return listfiles


def prepare_processed_data(task_name: str, listfile: pd.DataFrame, reader: ProcessedSetReader):
    # Legacy period length calculation may cut off legitimate data
    source_path = Path(SEMITEMP_DIR, "processed", task_name)
    target_path = Path(TEMP_DIR, "processed", task_name)
    write = DataSetWriter(target_path)
    X_processed, y_processed = reader.read_samples(read_ids=True).values()
    if "period_length" in listfile.columns:
        for subject_id in X_processed:
            for stay_ids in X_processed[subject_id]:
                X = X_processed[subject_id][stay_ids]
                X = X[X.index < listfile.loc[f"{subject_id}_episode{stay_ids}_timeseries.csv"]
                      ["period_length"] + 1e-6]
                X_processed[subject_id][stay_ids] = X

    write.write_bysubject({"X": X_processed})
    write.write_bysubject({"y": y_processed})

    for file in source_path.iterdir():
        if not file.is_file():
            continue
        shutil.copy(str(file), str(target_path))
    return


def assert_strategy_equals(X_strategy: dict, test_data_dir: Path):
    tested_stay = set()
    subject_count = 0
    stay_count = 0

    tests_io(f"Subjects tested: {subject_count}\n"
             f"Stays tested: {stay_count}")

    for test_file_path in test_data_dir.iterdir():
        test_df = pd.read_csv(test_file_path, na_values=[''], keep_default_na=False)
        test_df.index.name = "bins"

        # Extract subject id and stay id
        match = re.search(r"(\d+)_episode(\d+)_timeseries\.csv", test_file_path.name)
        subject_id = int(match.group(1))
        stay_id = int(match.group(2))
        tested_stay.add(stay_id)

        # Read sample
        subject_count += int(all([stay in tested_stay for stay in X_strategy[subject_id]]))
        stay_count += 1

        X = X_strategy[subject_id][stay_id]

        # Ensure column identity
        test_df.columns = test_df.columns.str.strip(" ")
        X.columns = X.columns.str.strip(" ")

        missing_columns = set(test_df) - set(X)
        additional_columns = set(X) - set(test_df)
        assert not missing_columns
        assert not additional_columns

        # Test X against ground truth
        test_df = test_df[X.columns]
        assert_dataframe_equals(X.astype(float), test_df)

        tests_io(f"Subjects tested: {subject_count}\n"
                 f"Stays tested: {stay_count}",
                 flush_block=True)
