import pytest
import datasets
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datasets.readers import ProcessedSetReader
from preprocessing.discretizers import BatchDiscretizer
from tests.utils.general import assert_dataframe_equals
from utils.IO import *
from tests.settings import *


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_discretizer(task_name: str, reader: ProcessedSetReader):
    # Path to discretizer sets
    test_data_dir = Path(TEST_GT_DIR, "discretized", TASK_NAME_MAPPING[task_name])
    # Listfile with truth values
    listfile = pd.read_csv(Path(test_data_dir, "listfile.csv")).set_index("stay")
    message_prelude = [f"Testing discretizer for task {task_name}"]
    stay_name_regex = r"(\d+)_episode(\d+)_timeseries\.csv"
    strategy_count = {"previous": 0, "next": 0, "zero": 0, "normal": 0}

    start_printing = True
    listfile = listfile.reset_index()
    listfile["subject"] = listfile["stay"].apply(lambda x: re.search(stay_name_regex, x).group(1))
    listfile["icustay"] = listfile["stay"].apply(lambda x: re.search(stay_name_regex, x).group(2))
    listfile = listfile.set_index("stay")

    n_subject = listfile["subject"].nunique()
    n_stay = listfile["icustay"].nunique()

    for curr_test_dir in test_data_dir.iterdir():
        if not curr_test_dir.is_dir():
            continue

        tested_stay = set()
        subject_count = 0
        stay_count = 0

        # One directory per setting
        match = re.search(r"imp(\w+)_start(\w+)", curr_test_dir.name)
        impute_strategy = match.group(1).replace("_value", "")  # previous, next, zero, normal
        start_strategy = match.group(2)  # zero, relative

        if start_printing:
            tests_io(
                "\n".join(message_prelude) +
                f"\nTesting discretizer for impute strategy '{impute_strategy}' \nwith start strategy '{start_strategy}'\n"
                f"Subjects tested: {subject_count}/{n_subject}\n"
                f"Stays tested: {stay_count}/{n_stay}\n")
            start_printing = False

        # Emulate discretizer settings
        discretizer = BatchDiscretizer(time_step_size=1.0,
                                       start_at_zero=(start_strategy == "zero"),
                                       impute_strategy=impute_strategy)

        for test_file_path in curr_test_dir.iterdir():
            test_df = pd.read_csv(test_file_path)

            # Extract subject id and stay id
            match = re.search(r"(\d+)_episode(\d+)_timeseries\.csv", test_file_path.name)
            subject_id = match.group(1)
            stay_id = match.group(2)
            tested_stay.add(stay_id)

            # Read sample
            X_subj, y_subj = reader.read_sample(subject_id, read_ids=True).values()
            subject_count += int(all([stay in tested_stay for stay in X_subj]))
            stay_count += 1

            X, y = X_subj[stay_id], y_subj[stay_id]

            # Legacy period length calculation may cut off data
            if "period_length" in listfile.columns:
                X = X[X.index < listfile.loc[test_file_path.name]["period_length"] + 1e-6]

            # Process the sample (equivalent to transform)
            imputed_df = X.copy()
            discretized_df = discretizer._categorize_data(imputed_df.copy())
            transformed_df = discretizer._bin_data(discretized_df.copy())
            transformed_df = discretizer._impute_data(transformed_df.copy())

            # Ensure column identity
            test_df.columns = test_df.columns.str.strip(" ")
            transformed_df.columns = transformed_df.columns.str.strip(" ")

            missing_columns = set(test_df) - set(transformed_df)
            additional_columns = set(transformed_df) - set(test_df)
            assert not missing_columns
            assert not additional_columns

            # Test X against ground truth
            test_df = test_df[transformed_df.columns]
            assert_dataframe_equals(transformed_df.astype(float), test_df)

            tests_io(
                "\n".join(message_prelude) +
                f"\nTesting discretizer for impute strategy '{impute_strategy}' with start strategy '{start_strategy}'\n"
                f"Subjects tested: {subject_count}/{n_subject}\n"
                f"Stays tested: {stay_count}/{n_stay}\n",
                flush_block=True)

        strategy_count[impute_strategy] += 1

        if strategy_count[impute_strategy] == 2:
            message_prelude.append(f"Done testing impute startegs '{impute_strategy}'")
            print("")


if __name__ == "__main__":

    for task_name in TASK_NAMES:
        reader = datasets.load_data(chunksize=75837,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=TEMP_DIR,
                                    preprocess=True,
                                    task=task_name)
        test_discretizer(task_name=task_name, reader=reader)
