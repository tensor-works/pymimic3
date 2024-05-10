import datasets
import shutil
import pytest
import random
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
from utils.IO import *
from tests.settings import *
from datasets.readers import ProcessedSetReader, SplitSetReader
from sklearn import model_selection
from pathlib import Path
from datasets.trackers import DataSplitTracker, PreprocessingTracker
from pathos.multiprocessing import Pool, cpu_count
from datasets.split import ReaderSplitter
from utils import dict_subset, is_numerical


@pytest.mark.parametrize("preprocessing_style", ["discretized", "engineered"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_ratio_split(
    task_name,
    preprocessing_style,
    discretized_readers: Dict[str, ProcessedSetReader],
    engineered_readers: Dict[str, ProcessedSetReader],
):
    tests_io(f"Test case by ratio for task: {task_name}", level=0)
    # Discretization or feature engineering
    if preprocessing_style == "discretized":
        reader = discretized_readers[task_name]
    else:
        reader = engineered_readers[task_name]

    tolerance = (1e-2 if task_name in ["DECOMP", "LOS"] else 1e-1)
    for test_size in [-0.1, 1.1]:
        with pytest.raises(ValueError):
            _: SplitSetReader = datasets.train_test_split(reader, test_size=test_size, val_size=0.2)
    for val_size in [-0.1, 1.1]:
        with pytest.raises(ValueError):
            _: SplitSetReader = datasets.train_test_split(reader, test_size=0.1, val_size=val_size)

    reader = discretized_readers[task_name]
    all_split_subjects = dict()
    curr_iter = 0

    # Test splitting for a few values
    for test_size in [0.0, 0.2, 0.4]:
        all_split_subjects[test_size] = dict()
        for val_size in [0.0, 0.2, 0.4]:
            all_split_subjects[test_size][val_size] = assert_split_sanity(
                test_size, val_size, tolerance, reader,
                datasets.train_test_split(reader,
                                          test_size=test_size,
                                          val_size=val_size,
                                          storage_path=Path(TEMP_DIR, "split_test",
                                                            str(curr_iter))))
            curr_iter += 1

    # Test restoring splits for previous values
    for test_size in [0.0, 0.2, 0.4]:
        for val_size in [0.0, 0.2, 0.4]:
            split_reader = datasets.train_test_split(reader,
                                                     test_size=test_size,
                                                     val_size=val_size,
                                                     storage_path=Path(
                                                         TEMP_DIR, "split_test", str(curr_iter)))
            for split_name in all_split_subjects[test_size][val_size]:
                assert set(all_split_subjects[test_size][val_size][split_name]) == \
                    set(getattr(split_reader, split_name).subject_ids)

            curr_iter += 1


@pytest.mark.parametrize("preprocessing_style", ["discretized", "engineered"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_demographic_filter(
    task_name,
    preprocessing_style,
    discretized_readers: Dict[str, ProcessedSetReader],
    engineered_readers: Dict[str, ProcessedSetReader],
):
    tests_io("Test case by demographic filter", level=0)
    # Discretization or feature engineering
    if preprocessing_style == "discretized":
        reader = discretized_readers[task_name]
    else:
        reader = engineered_readers[task_name]

    # Test on unknown demographic
    with pytest.raises(ValueError):
        demographic_filter = {"INVALID": {"less": 0.5}}
        _: SplitSetReader = datasets.train_test_split(reader, demographic_filter=demographic_filter)

    # Test on invalid range
    for less_key in ["less", "leq"]:
        for greater_key in ["greater", "geq"]:
            with pytest.raises(ValueError):
                demographic_filter = {"AGE": {less_key: 10, greater_key: 90}}
                _: SplitSetReader = datasets.train_test_split(reader,
                                                              demographic_filter=demographic_filter)

    subject_info_df = pd.read_csv(Path(reader.root_path, "subject_info.csv"))
    subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(reader.subject_ids)]
    curr_iter = 0
    all_filters = list()
    for attribute in set(subject_info_df.columns) - set(["SUBJECT_ID", "ICUSTAY_ID"]):
        attribute_data = subject_info_df[attribute]
        if is_numerical(attribute_data.to_frame()):
            # Test on all range key words
            sample_filters = sample_numeric_filter(attribute, attribute_data)
            all_filters.extend(sample_filters)
            for demographic_filter in sample_filters:
                check_numerical_attribute(attribute, demographic_filter, subject_info_df, curr_iter)
        else:
            # Categorical column
            demographic_filter = sample_categorical_filter(attribute, attribute_data)
            all_filters.append(demographic_filter)
            check_categorical_attribute(attribute, demographic_filter, subject_info_df, curr_iter)
        curr_iter += 1

    for _ in range(10):
        demographic_filter = sample_hetero_filter(all_filters, 3)
        check_hetero_attributes(attribute, demographic_filter, subject_info_df, curr_iter)
        curr_iter += 1


def test_demographic_split():
    subject_info_df = pd.read_csv(Path(reader.root_path, "subject_info.csv"))
    subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(reader.subject_ids)]
    curr_iter = 0
    all_filters = list()
    for attribute in set(subject_info_df.columns) - set(["SUBJECT_ID", "ICUSTAY_ID"]):
        if is_numerical(subject_info_df[attribute].to_frame()):
            # Test on all range key words
            possible_filter = sample_numeric_filter(attribute, subject_info_df[attribute])
            all_filters.extend(possible_filter)
        else:
            # Categorical column
            demographic_filter = sample_categorical_filter(attribute, subject_info_df[attribute])
            all_filters.append(demographic_filter)
        curr_iter += 1

    for _ in range(10):
        demographic_filter = sample_hetero_filter(all_filters, 1)
        reader: SplitSetReader = datasets.train_test_split(
            reader, demographic_split={"test": demographic_filter})
        curr_iter += 1


def sample_categorical_filter(attribute: str, attribute_data: pd.Series):
    # Randomly choose half of the possible categories
    categories = attribute_data.unique()
    choices = random.sample(categories.tolist(), k=int(np.floor(len(categories) / 2)))
    # Demographic filter
    demographic_filter = {attribute: {"choice": choices}}
    return demographic_filter


def sample_hetero_filter(all_filters: list, num: int):
    # Test on random demographic
    demographic_filter = dict()
    filter_selection = random.sample(all_filters, k=max(num * 5, len(all_filters)))
    attributes = list()
    for curr_filter in filter_selection:
        if not set(curr_filter.keys()) & set(attributes):
            if list(curr_filter.values())[0]:
                demographic_filter.update(curr_filter)
                attributes.extend(curr_filter.keys())
        if len(demographic_filter) == num:
            break
    return demographic_filter


def assert_split_sanity(test_size: float, val_size: float, tolerance: float,
                        reader: ProcessedSetReader, split_reader: SplitSetReader):
    tracker = PreprocessingTracker(Path(reader.root_path, "progress"))
    subject_counts = tracker.subjects
    subject_ids = tracker.subject_ids
    split_samples = dict()
    split_subjects = dict()

    # Extract result for each split
    for split_name in split_reader.split_names:
        # Check sample is not empty
        random_sample = getattr(split_reader, split_name).random_samples()
        assert len(random_sample["X"])
        # Get subject IDs and sample counts
        split_ids = getattr(split_reader, split_name).subject_ids
        split_subjects[split_name] = split_ids
        split_samples[split_name] = sum([
            stay_counts["total"]
            for stay_counts in dict_subset(subject_counts, split_ids).values()
            if isinstance(stay_counts, dict)
        ])

    # Assert no duplicte subjects
    assert not set().intersection(*split_subjects.values())
    # Assert ratios are respected
    total_samples = sum(split_samples.values())
    assert sum([
        len(getattr(split_reader, split_name).subject_ids)
        for split_name in split_reader.split_names
    ]) == len(subject_ids)
    assert abs(split_samples["train"] / total_samples - (1 - test_size - val_size)) < tolerance
    if test_size:
        assert abs(split_samples["test"] / total_samples - test_size) < tolerance
    if val_size:
        assert abs(split_samples["val"] / total_samples - val_size) < tolerance

    return split_subjects


def check_hetero_attributes(attribute, demographic_filter, subject_info_df, curr_iter):
    split_reader: SplitSetReader = datasets.train_test_split(reader,
                                                             demographic_filter=demographic_filter,
                                                             storage_path=Path(
                                                                 TEMP_DIR, str(curr_iter)))
    if split_reader.train is not None:
        for attribute in demographic_filter:
            attribute_data = subject_info_df[attribute]
            if is_numerical(attribute_data.to_frame()):
                if attribute in subject_info_df.columns:
                    selected_attributes = subject_info_df[subject_info_df["SUBJECT_ID"].isin(
                        split_reader.train.subject_ids)][attribute]
                    assert_range(selected_attributes, demographic_filter[attribute])
            else:
                selected_attributes = subject_info_df[subject_info_df["SUBJECT_ID"].isin(
                    split_reader.train.subject_ids)][attribute]
                assert set(selected_attributes).issubset(demographic_filter[attribute]["choice"])


def check_categorical_attribute(attribute, demographic_filter, subject_info_df, curr_iter):
    # Apply
    split_reader: SplitSetReader = datasets.train_test_split(reader,
                                                             demographic_filter=demographic_filter,
                                                             storage_path=Path(
                                                                 TEMP_DIR, str(curr_iter)))

    selected_attributes = subject_info_df[subject_info_df["SUBJECT_ID"].isin(
        split_reader.train.subject_ids)][attribute]

    # Ensure all entries are from choices. Subjects with stays where the attribute is from selected and not selected categories
    # are not included in the demographic.
    assert not set(selected_attributes) - set(subject_info_df[attribute])


def check_numerical_attribute(attribute, demographic_filter, subject_info_df, curr_iter):
    # Make the split
    split_reader: SplitSetReader = datasets.train_test_split(reader,
                                                             demographic_filter=demographic_filter,
                                                             storage_path=Path(
                                                                 TEMP_DIR, str(curr_iter)))

    if split_reader.train is not None:
        selected_attributes = subject_info_df[subject_info_df["SUBJECT_ID"].isin(
            split_reader.train.subject_ids)][attribute]
        assert_range(selected_attributes, demographic_filter[attribute])  #


def assert_range(column, demographic_filter, invert=False):
    """
    Asserts that all or not any elements in the column meet the specified range conditions.
    
    Parameters:
        column (iterable): The input series or array-like structure to check.
        demographic_filter (dict): A dictionary containing range conditions like 'less', 'leq', 'greater', and 'geq'.
        invert (bool): If True, use 'not any' logic; otherwise, use 'all' logic.
    """

    def range_condition(val):
        # Define a function that checks the conditions specified in demographic_filter
        if "less" in demographic_filter:
            if "greater" in demographic_filter:
                return (val < demographic_filter["less"]) and (val > demographic_filter["greater"])
            elif "geq" in demographic_filter:
                return (val < demographic_filter["less"]) and (val >= demographic_filter["geq"])
            else:
                return val < demographic_filter["less"]
        elif "leq" in demographic_filter:
            if "greater" in demographic_filter:
                return (val <= demographic_filter["leq"]) and (val > demographic_filter["greater"])
            elif "geq" in demographic_filter:
                return (val <= demographic_filter["leq"]) and (val >= demographic_filter["geq"])
            else:
                return val <= demographic_filter["leq"]
        elif "greater" in demographic_filter:
            return val > demographic_filter["greater"]
        elif "geq" in demographic_filter:
            return val >= demographic_filter["geq"]
        return True

    # Apply the appropriate assertion logic
    if invert:
        assert not any(
            range_condition(x)
            for x in column), "Condition failed: Some elements meet the specified range condition"
    else:
        assert all(range_condition(x) for x in column
                  ), "Condition failed: Not all elements meet the specified range condition"


def sample_numeric_filter(column_name: str, column_sr: pd.Series):
    filters = list()
    for less_key in ["less", "leq", None]:
        for greater_key in ["greater", "geq", None]:
            # Get a valid range
            curr_range = column_sr.max() - column_sr.min()
            curr_filter = dict()
            if greater_key is not None:
                lower_bound = random.uniform(column_sr.min(), column_sr.min() + curr_range / 2)
                curr_filter[greater_key] = lower_bound
            else:
                lower_bound = column_sr.min()
            if less_key is not None:
                upper_bound = random.uniform(lower_bound + curr_range / 10, column_sr.max())
                curr_filter[less_key] = upper_bound
            else:
                upper_bound = column_sr.max()
            filters.append({column_name: curr_filter})
    return filters


def test_demographic_split():
    ...


if __name__ == "__main__":
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task='DECOMP')
    demographic_filter = {
        "DISCHARGE_LOCATION": {
            'choice': [
                'REHAB/DISTINCT PART HOSP', 'HOME HEALTH CARE', 'DEAD/EXPIRED',
                'HOME WITH HOME IV PROVIDR'
            ]
        }
    }
    split_reader = datasets.train_test_split(reader, demographic_filter=demographic_filter)
    subject_info_df = pd.read_csv(Path(reader.root_path, "subject_info.csv"))
    subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(
        split_reader.train.subject_ids)]
    subject_info_df.to_csv("trash.csv")
    test_demographic_filter("DECOMP", "discretized", {"DECOMP": reader}, {})
    discretized_readers = dict()
    engineered_readers = dict()
    if TEMP_DIR.is_dir():
        shutil.rmtree(TEMP_DIR)
    for task_name in TASK_NAMES:
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    discretize=True,
                                    time_step_size=1.0,
                                    start_at_zero=True,
                                    impute_strategy='previous',
                                    task=task_name)
        discretized_readers[task_name] = reader
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    engineer=True,
                                    task=task_name)
        engineered_readers[task_name] = reader
        for processing_style in ["discretized", "engineered"]:
            test_ratio_split(task_name, processing_style, discretized_readers, engineered_readers)
    pass
