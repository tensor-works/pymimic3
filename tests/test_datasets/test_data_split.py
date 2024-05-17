"""
TODO! Add current iter and test restoral
"""

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
from datasets.split.splitters import ReaderSplitter
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
            for demographic_filter in sample_filters:
                split_reader: SplitSetReader = datasets.train_test_split(
                    reader,
                    demographic_filter=demographic_filter,
                    storage_path=Path(TEMP_DIR, str(curr_iter)))
                if "train" in split_reader.split_names:
                    check_numerical_attribute(attribute=attribute,
                                              demographic_filter=demographic_filter,
                                              subject_info_df=subject_info_df,
                                              subject_ids=split_reader.train.subject_ids)
            all_filters.extend(sample_filters)
        else:
            # Categorical column
            demographic_filter = sample_categorical_filter(attribute, attribute_data)
            split_reader: SplitSetReader = datasets.train_test_split(
                reader,
                demographic_filter=demographic_filter,
                storage_path=Path(TEMP_DIR, str(curr_iter)))
            if "train" in split_reader.split_names:
                check_categorical_attribute(attribute=attribute,
                                            demographic_filter=demographic_filter,
                                            subject_info_df=subject_info_df,
                                            subject_ids=split_reader.train.subject_ids)
            all_filters.append(demographic_filter)
        curr_iter += 1

    for _ in range(10):
        demographic_filter = sample_hetero_filter(3, subject_info_df)
        split_reader: SplitSetReader = datasets.train_test_split(
            reader,
            demographic_filter=demographic_filter,
            storage_path=Path(TEMP_DIR, str(curr_iter)))
        if "train" in split_reader.split_names:
            check_hetero_attributes(demographic_filter=demographic_filter,
                                    subject_info_df=subject_info_df,
                                    subject_ids=split_reader.train.subject_ids)
        curr_iter += 1


def test_demo_split(
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

    subject_info_df = pd.read_csv(Path(reader.root_path, "subject_info.csv"))
    subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(reader.subject_ids)]
    curr_iter = 0

    for _ in range(10):
        demographic_filter = sample_hetero_filter(1, subject_info_df)

        split_reader: SplitSetReader = datasets.train_test_split(
            reader, demographic_split={"test": demographic_filter})
        if "test" in split_reader.split_names:
            check_hetero_attributes(demographic_filter=demographic_filter,
                                    subject_info_df=subject_info_df,
                                    subject_ids=split_reader.test.subject_ids)

            assert not set(split_reader.test.subject_ids) & set(split_reader.train.subject_ids)
        #TODO! Add current iter and test restoral
        tests_io(f"Succeeded testing the filter for test!")

    for _ in range(10):
        test_filter, val_filter = sample_hetero_filter(1, subject_info_df, val_set=True)
        #TODO! Add current iter and test restoral
        split_reader: SplitSetReader = datasets.train_test_split(reader,
                                                                 demographic_split={
                                                                     "test": test_filter,
                                                                     "val": val_filter
                                                                 })
        if "test" in split_reader.split_names:
            check_hetero_attributes(demographic_filter=test_filter,
                                    subject_info_df=subject_info_df,
                                    subject_ids=split_reader.test.subject_ids)
            if "train" in split_reader.split_names:
                assert not set(split_reader.test.subject_ids) & set(split_reader.train.subject_ids)

        if "val" in split_reader.split_names:
            check_hetero_attributes(demographic_filter=val_filter,
                                    subject_info_df=subject_info_df,
                                    subject_ids=split_reader.val.subject_ids)
            if "test" in split_reader.split_names:
                assert not set(split_reader.val.subject_ids) & set(split_reader.train.subject_ids)
        if "val" in split_reader.split_names and "test" in split_reader.split_names:
            assert not set(split_reader.test.subject_ids) & set(split_reader.val.subject_ids)

        curr_iter += 1
        tests_io(f"Succeeded testing the filter for val and test!")


def test_demo_and_ratio_split(
    task_name,
    preprocessing_style,
    discretized_readers: Dict[str, ProcessedSetReader],
    engineered_readers: Dict[str, ProcessedSetReader],
):
    tests_io("Test case demographic split with specified ratios", level=0)
    # Discretization or feature engineering
    if preprocessing_style == "discretized":
        reader = discretized_readers[task_name]
    else:
        reader = engineered_readers[task_name]

    subject_info_df = pd.read_csv(Path(reader.root_path, "subject_info.csv"))
    subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(reader.subject_ids)]
    attribute = random.choice(["WARDID", "AGE"])
    attribute_data = subject_info_df[attribute]
    filters = sample_numeric_filter(attribute, attribute_data)

    tolerance = (1e-2 if task_name in ["DECOMP", "LOS"] else 1e-1)
    test_size = 0.2
    none_reader = 0

    for demographic_filter in filters:
        tests_io(f"Specified demographic filter: 'test': {demographic_filter}")
        if not all([value for value in demographic_filter.values()]):
            continue
        try:
            split_reader: SplitSetReader = datasets.train_test_split(
                reader, 0.2, demographic_split={"test": demographic_filter})
        except ValueError:
            none_reader += 1
            continue

        if "test" in split_reader.split_names and "train" in split_reader.split_names:
            check_hetero_attributes(demographic_filter=demographic_filter,
                                    subject_info_df=subject_info_df,
                                    subject_ids=split_reader.test.subject_ids)

            assert_split_sanity(test_size, 0, tolerance, reader, split_reader, reduced_set=True)
        else:
            none_reader += 1
    if none_reader == len(filters):
        tests_io(f"All splits have invalid sets for {attribute}!")

    filters = sample_numeric_filter(attribute, attribute_data, val_set=True)

    split_size = 0.2
    none_reader = 0
    for test_filter, val_filter in filters:
        tests_io(f"Specified demographic filter: 'test': {test_filter}, 'val': {val_filter}")

        split_filters = {"test": test_filter, "val": val_filter}
        try:
            split_reader: SplitSetReader = datasets.train_test_split(
                reader, split_size, split_size, demographic_split=split_filters)
        except ValueError as e:
            none_reader += 1
            continue
        if len(split_reader.split_names) > 2:
            for set_name in split_reader.split_names:
                if set_name in ["test", "val"]:
                    check_hetero_attributes(demographic_filter=split_filters[set_name],
                                            subject_info_df=subject_info_df,
                                            subject_ids=getattr(split_reader, set_name).subject_ids)

            assert_split_sanity(split_size,
                                split_size,
                                tolerance,
                                reader,
                                split_reader,
                                reduced_set=True)

        else:
            none_reader += 1

    if none_reader == len(filters):
        tests_io(f"All splits have invalid sets for {attribute}!")


def test_train_size(task_name, preprocessing_style, discretized_readers: Dict[str,
                                                                              ProcessedSetReader],
                    engineered_readers: Dict[str, ProcessedSetReader]):

    tests_io("Test case train size", level=0)
    # Discretization or feature engineering
    if preprocessing_style == "discretized":
        reader = discretized_readers[task_name]
    else:
        reader = engineered_readers[task_name]

    # Train size with ratio
    tolerance = (1e-2 if task_name in ["DECOMP", "LOS"] else 1e-1)

    curr_iter = 0
    write_bool = True
    train_size = 8

    for val_size in [0.0, 0.2]:
        if val_size and write_bool:

            write_bool = False
        for test_size in [0.2, 0.4]:
            tests_io(
                f"Specified train_size: {train_size}; test_size: {test_size}; val_size: {val_size}")
            split_reader: SplitSetReader = \
                datasets.train_test_split(reader,
                                          test_size=test_size,
                                          val_size=val_size,
                                          storage_path=Path(
                                              TEMP_DIR, "split_test",
                                              str(curr_iter) + "base"))

            test_split_reader: SplitSetReader = \
                datasets.train_test_split(reader,
                                          test_size=test_size,
                                          val_size=val_size,
                                          train_size=train_size,
                                          storage_path=Path(TEMP_DIR, "split_test",
                                          str(curr_iter) + "_test"))
            if len(split_reader.train.subject_ids) < train_size:
                assert len(test_split_reader.train.subject_ids) == len(
                    split_reader.train.subject_ids)
            else:
                assert len(test_split_reader.train.subject_ids) == train_size
            assert_split_sanity(test_size,
                                val_size,
                                tolerance,
                                reader,
                                test_split_reader,
                                reduced_set=True)
            curr_iter += 1

    # Train size with demographic split
    tests_io("Specific train size with demographic split")
    subject_info_df = pd.read_csv(Path(reader.root_path, "subject_info.csv"))
    subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(reader.subject_ids)]
    attribute = random.choice(["WARDID", "AGE"])
    attribute_data = subject_info_df[attribute]
    filters = sample_numeric_filter(attribute, attribute_data)

    test_size = 0.2
    val_size = 0.2
    none_reader = 0

    for demographic_filter in filters:
        print(demographic_filter)
        try:
            split_reader: SplitSetReader = datasets.train_test_split(
                reader,
                test_size,
                val_size,
                train_size=train_size,
                demographic_split={"test": demographic_filter})
        except ValueError:
            none_reader += 1
            continue

        if not set(["test", "train", "val"]) - set(split_reader.split_names):
            check_hetero_attributes(demographic_filter=demographic_filter,
                                    subject_info_df=subject_info_df,
                                    subject_ids=split_reader.test.subject_ids)

            assert_split_sanity(test_size,
                                val_size,
                                tolerance,
                                reader,
                                split_reader,
                                reduced_set=True)
        else:
            none_reader += 1
    if none_reader == len(filters):
        tests_io(f"All splits have invalid sets for {attribute}!")


def sample_categorical_filter(attribute: str, attribute_sr: pd.Series, val_set: bool = False):
    # Randomly choose half of the possible categories
    categories = attribute_sr.unique()
    test_choices = random.sample(categories.tolist(), k=int(np.floor(len(categories) / 2)))
    # Demographic filter
    if val_set and len(test_choices):
        val_choices = random.sample(test_choices, k=int(len(test_choices) / 2))
        test_choices = list(set(test_choices) - set(val_choices))
        if test_choices and val_choices:
            return {attribute: {"choice": test_choices}}, {attribute: {"choice": val_choices}}
        else:
            return {}, {}

    return {attribute: {"choice": test_choices}}


def sample_numeric_filter(attribute: str, attribute_sr: pd.Series, val_set: bool = False):

    def get_range_key(greater_key: str,
                      less_key: str,
                      min_value: int = -1e10,
                      max_value: int = -1e10):
        curr_filter = dict()
        curr_range = max_value - min_value
        if greater_key is not None:
            lower_bound = random.uniform(min_value, min_value + curr_range / 2)
            curr_filter[greater_key] = lower_bound
        else:
            lower_bound = min_value
        if less_key is not None:
            upper_bound = random.uniform(lower_bound + curr_range / 10, max_value)
            curr_filter[less_key] = upper_bound
        else:
            upper_bound = max_value
        return curr_filter

    filters = list()
    inversion_mapping = {"less": "greater", "leq": "geq", "greater": "less", "geq": "leq"}
    for less_key in ["less", "leq", None]:
        for greater_key in ["greater", "geq", None]:
            # Get a valid range
            if less_key is None and greater_key is None:
                continue
            test_filter = get_range_key(greater_key, less_key, attribute_sr.min(),
                                        attribute_sr.max())
            if val_set:
                if test_filter.get(less_key) is None:
                    if test_filter.get(greater_key) is None:
                        # Both are None
                        continue
                    # Only less is not None
                    val_filter = get_range_key(greater_key,
                                               inversion_mapping[greater_key],
                                               min_value=attribute_sr.min(),
                                               max_value=test_filter[greater_key])
                else:
                    if test_filter.get(greater_key) is None:
                        # Only greater is not None
                        val_filter = get_range_key(inversion_mapping[less_key],
                                                   less_key,
                                                   min_value=test_filter[less_key],
                                                   max_value=attribute_sr.max())
                    else:
                        # Both are set
                        if random.choice([True, False]):
                            val_filter = get_range_key(greater_key,
                                                       less_key,
                                                       min_value=test_filter[less_key],
                                                       max_value=attribute_sr.max())
                        else:
                            val_filter = get_range_key(greater_key,
                                                       less_key,
                                                       min_value=attribute_sr.min(),
                                                       max_value=test_filter[greater_key])
                filters.append(({attribute: test_filter}, {attribute: val_filter}))
            else:
                filters.append({attribute: test_filter})
    return filters


def sample_hetero_filter(num: int, subject_info_df: pd.DataFrame, val_set: bool = False):
    # Test on random demographic
    all_filters = list()
    for attribute in set(subject_info_df.columns) - set(["SUBJECT_ID", "ICUSTAY_ID"]):
        if is_numerical(subject_info_df[attribute].to_frame()):
            # Test on all range key words
            possible_filter = sample_numeric_filter(attribute=attribute,
                                                    attribute_sr=subject_info_df[attribute],
                                                    val_set=val_set)
            all_filters.extend(possible_filter)
        else:
            # Categorical column
            possible_filter = sample_categorical_filter(attribute=attribute,
                                                        attribute_sr=subject_info_df[attribute],
                                                        val_set=val_set)
            all_filters.append(possible_filter)
    test_filter = dict()
    val_filter = dict()
    filter_selection = random.sample(all_filters, k=max(num * 5, len(all_filters)))
    attributes = list()
    for curr_filter in filter_selection:
        # No duplicate for same attribute
        if val_set and not set(curr_filter[0].keys()) & set(attributes):
            # No empty filters
            if val_set and list(curr_filter[0].values())[0] and list(curr_filter[1].values())[0]:
                test_filter.update(curr_filter[0])
                val_filter.update(curr_filter[1])
                attributes.extend(curr_filter[0].keys())
        elif not val_set and not set(curr_filter.keys()) & set(attributes):
            # No empty filters
            if not val_set and list(curr_filter.values())[0]:
                test_filter.update(curr_filter)
                attributes.extend(curr_filter.keys())
        if len(test_filter) == num:
            break
    if val_set:
        return test_filter, val_filter
    return test_filter


def assert_split_sanity(test_size: float,
                        val_size: float,
                        tolerance: float,
                        reader: ProcessedSetReader,
                        split_reader: SplitSetReader,
                        reduced_set: bool = False):
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
    # Assert ratios are respected
    # Assert no duplicte subjects
    assert not set().intersection(*split_subjects.values())
    if not reduced_set:
        assert sum([
            len(getattr(split_reader, split_name).subject_ids)
            for split_name in split_reader.split_names
        ]) == len(subject_ids)
    check_split_sizes(split_samples, test_size, val_size, tolerance)
    return split_subjects


def check_split_sizes(split_samples: dict, test_size: float, val_size: float, tolerance: float):
    total_samples = sum(split_samples.values())
    assert abs(split_samples["train"] / total_samples - (1 - test_size - val_size)) < tolerance

    if test_size:
        assert abs(split_samples["test"] / total_samples - test_size) < tolerance

    if val_size:
        assert abs(split_samples["val"] / total_samples - val_size) < tolerance


def check_hetero_attributes(demographic_filter: dict, subject_info_df: pd.DataFrame,
                            subject_ids: List[int]):

    for attribute in demographic_filter:
        attribute_data = subject_info_df[attribute]
        if is_numerical(attribute_data.to_frame()):
            check_numerical_attribute(attribute=attribute,
                                      demographic_filter=demographic_filter,
                                      subject_info_df=subject_info_df,
                                      subject_ids=subject_ids)
        else:
            check_categorical_attribute(attribute=attribute,
                                        demographic_filter=demographic_filter,
                                        subject_info_df=subject_info_df,
                                        subject_ids=subject_ids)


def check_categorical_attribute(attribute: str, demographic_filter: dict,
                                subject_info_df: pd.DataFrame, subject_ids: List[int]):
    selected_attributes = subject_info_df[subject_info_df["SUBJECT_ID"].isin(
        subject_ids)][attribute]
    # Ensure all entries are from choices. Subjects with stays where the attribute is from selected and not selected categories
    # are not included in the demographic.
    assert not set(selected_attributes) - set(demographic_filter[attribute]["choice"])


def check_numerical_attribute(attribute: str, demographic_filter: dict,
                              subject_info_df: pd.DataFrame, subject_ids: List[int]):
    # Make the split
    selected_attributes = subject_info_df[subject_info_df["SUBJECT_ID"].isin(
        subject_ids)][attribute]
    assert_range(selected_attributes, demographic_filter[attribute])


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


if __name__ == "__main__":
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task='DECOMP')
    # test_demographic_filter("DECOMP", "discretized", {"DECOMP": reader}, {})
    # test_demo_split("DECOMP", "discretized", {"DECOMP": reader}, {})
    # test_demo_and_ratio_split("DECOMP", "discretized", {"DECOMP": reader}, {})

    # for _ in range(10):
    #     test_demo_and_ratio_split("DECOMP", "discretized", {"DECOMP": reader}, {})
    # test_train_size("DECOMP", "discretized", {"DECOMP": reader}, {})

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
            test_demographic_filter(task_name, processing_style, discretized_readers,
                                    engineered_readers)
            test_demo_split(task_name, processing_style, discretized_readers, engineered_readers)
            test_demo_and_ratio_split(task_name, processing_style, discretized_readers,
                                      engineered_readers)
            test_train_size(task_name, processing_style, discretized_readers, engineered_readers)
            test_ratio_split(task_name, processing_style, discretized_readers, engineered_readers)
    pass
