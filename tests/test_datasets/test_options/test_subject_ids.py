import datasets
import random
import shutil
import pytest
import re
import pandas as pd
from copy import deepcopy
from pathlib import Path
from utils.IO import *
from tests.settings import *
from tests.pytest_utils import copy_dataset
from tests.pytest_utils.general import assert_dataframe_equals
from tests.pytest_utils.preprocessing import assert_reader_equals, assert_dataset_equals
from tests.pytest_utils.feature_engineering import extract_test_ids, concatenate_dataset
from tests.pytest_utils.extraction import compare_extracted_datasets
from datasets.readers import ExtractedSetReader, ProcessedSetReader


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
def test_subject_ids_extraction(extracted_reader: ExtractedSetReader, subject_ids: list,
                                extraction_style: str):
    # Comapring subject event dfs for correctniss is costly due to unorderedness
    # These are subjects with a very low subject event count
    tests_io(f"Test case subject ids for extraction", level=0)

    # Retrive subject ids
    subjects = deepcopy(subject_ids)
    curr_subjects = list()
    # Sample subjects and test result by extending the subject id list
    for num_subjects in [1, 10, 10]:
        # Extend the subject id list
        curr_subjects.extend(random.sample(subjects, num_subjects - len(curr_subjects)))
        subjects = list(set(subjects) - set(curr_subjects))

        # Compare the extracted data with the test data
        tests_io(f"-> Testing extract-only with {len(curr_subjects)} subjects on empty directory.\n"
                 f"Subject IDs: {*curr_subjects,}")

        extract_and_compare(subject_ids=curr_subjects,
                            test_subject_ids=extracted_reader.subject_ids,
                            extraction_style=extraction_style,
                            extracted_reader=extracted_reader)

    tests_io(f"-> Succeeded in testing extending subject id.")

    # Sample subject from scratch and extract into empty dir
    extracted_dir = Path(TEMP_DIR, "extracted")
    for num_subjects in [11, 21]:
        # Remove the extracted directory
        if extracted_dir.is_dir():
            shutil.rmtree(str(extracted_dir))
        # Sample subject ids
        curr_subjects = random.sample(subjects, num_subjects)
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare the extracted data with the test data
        tests_io(f"-> Testing extract-only with {len(curr_subjects)} subjects on empty directory.\n"
                 f"Subject IDs: {*curr_subjects,}")
        extract_and_compare(subject_ids=curr_subjects,
                            test_subject_ids=extracted_reader.subject_ids,
                            extraction_style=extraction_style,
                            extracted_reader=extracted_reader)

    tests_io(f"-> Succeeded in testing on empty directory.")

    # Test reducing subject count to a single subject
    curr_subjects = random.sample(subjects, 1)
    extract_and_compare(subject_ids=curr_subjects,
                        test_subject_ids=extracted_reader.subject_ids,
                        extraction_style=extraction_style,
                        extracted_reader=extracted_reader)

    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_subject_ids_preprocessing_only(task_name: str, subject_ids: list, extraction_style: str):
    tests_io(f"Test case subject ids for preprocessing-only for task {task_name}.", level=0)
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Make sure extracted data is available
    tests_io(f"Preparing fully extracted directory")
    copy_dataset("extracted")

    # Retrive subject ids
    curr_subjects = list()
    subjects = deepcopy(subject_ids)

    # Sample subjects and test result by extending the subject id list
    for num_subjects in [1, 10, 10]:
        # Extend the subject id list
        curr_subjects.extend(random.sample(subjects, num_subjects - len(curr_subjects)))
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare generated results
        tests_io(f"-> Testing preprocessing-only with {len(curr_subjects)}"
                 " subjects on existing directory.")
        process_and_compare(subject_ids=curr_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Sample subject from scratch and extract into empty dir
    processed_dir = Path(TEMP_DIR, "processed")

    for num_subjects in [11, 21]:
        # Remove the processed directory
        if processed_dir.is_dir():
            shutil.rmtree(str(processed_dir))
        # Sample subject ids
        curr_subjects = random.sample(subjects, num_subjects)
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare the processed data
        tests_io(f"-> Testing preprocessing-only with {len(curr_subjects)}"
                 " subjects on empty directory.")
        process_and_compare(subject_ids=curr_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on empty directory.")
    # Test reducing subject count
    curr_subjects = random.sample(subjects, 1)
    process_and_compare(subject_ids=curr_subjects,
                        task_name=task_name,
                        extraction_style=extraction_style,
                        test_data_dir=test_data_dir)
    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_subject_ids_engineer_only(task_name: str, subject_ids: list, extraction_style: str):
    tests_io(f"Test case subject ids for engineering-only for task {task_name}.", level=0)

    # Test only engineering
    tests_io(f"Preparing fully preprocessed directory")
    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))

    test_data_dir = Path(TEST_GT_DIR, "engineered",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Load test data and extract ids
    test_df = pd.read_csv(Path(test_data_dir, "X.csv"))
    test_df = extract_test_ids(test_df)

    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))
    # Retrive subject ids
    curr_subjects = list()
    subjects = deepcopy(subject_ids)

    # Align unstructured frames
    test_df = test_df.sort_values(by=test_df.columns.to_list())
    test_df = test_df.reset_index(drop=True)

    # Test on existing directory
    for num_subjects in [1, 10, 10]:
        # Extend the subject id list
        curr_subjects.extend(random.sample(subjects, num_subjects - len(curr_subjects)))
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare generated results
        tests_io(
            f"-> Testing engineer-only with {len(curr_subjects)} subjects on existing directory.")
        engineer_and_compare(subject_ids=curr_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Test on empty directory
    engineered_dir = Path(TEMP_DIR, "engineered")
    for num_subjects in [11, 16]:
        if TEMP_DIR.is_dir():
            shutil.rmtree(engineered_dir)
        # Sample subject ids
        curr_subjects = random.sample(subjects, num_subjects)
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare the processed data
        tests_io(f"-> Testing engineer-only with {len(curr_subjects)} subjects on empty directory.")
        engineer_and_compare(subject_ids=curr_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on empty directory.")

    # Test reducing subject count
    curr_subjects = random.sample(subjects, 1)
    engineer_and_compare(subject_ids=curr_subjects,
                         task_name=task_name,
                         extraction_style=extraction_style,
                         test_df=test_df)
    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_subject_ids_preprocessing(task_name: str, subject_ids: list, extraction_style: str):
    # Test preprocessing from scratch, that means no fully extracted data in temp beforehand
    tests_io(f"Test case subject ids for preprocessing from scratch for task {task_name}.", level=0)
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Make sure extracted data is available
    tests_io(f"Preparing fully extracted directory")

    # Retrive subject ids
    curr_subjects = list()
    subjects = deepcopy(subject_ids)

    # Sample subjects and test result by extending the subject id list
    for num_subjects in [1, 10, 10]:
        # Extend the subject id list
        curr_subjects.extend(random.sample(subjects, num_subjects - len(curr_subjects)))
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare generated results
        tests_io(f"-> Testing preprocessing-only with {len(curr_subjects)}"
                 " subjects on existing directory.")
        process_and_compare(subject_ids=curr_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Sample subject from scratch and extract into empty dir
    processed_dir = Path(TEMP_DIR, "processed")

    for num_subjects in [11, 21]:
        # Remove the processed directory
        if processed_dir.is_dir():
            shutil.rmtree(str(processed_dir))
        # Sample subject ids
        curr_subjects = random.sample(subjects, num_subjects)
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare the processed data
        tests_io(f"-> Testing preprocessing-only with {len(curr_subjects)}"
                 " subjects on empty directory.")
        process_and_compare(subject_ids=curr_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on empty directory.")
    # Test reducing subject count
    curr_subjects = random.sample(subjects, 1)
    process_and_compare(subject_ids=curr_subjects,
                        task_name=task_name,
                        extraction_style=extraction_style,
                        test_data_dir=test_data_dir)
    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_subject_ids_engineer(task_name: str, subject_ids: list, extraction_style: str):
    tests_io(f"Test case subject ids for engineering from scratch for task {task_name}.", level=0)
    # Test engineering from scratch, that means no fully processed data in temp
    tests_io(f"Preparing fully preprocessed directory")
    test_data_dir = Path(TEST_GT_DIR, "engineered",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Load test data and extract ids
    test_df = pd.read_csv(Path(test_data_dir, "X.csv"))
    test_df = extract_test_ids(test_df)

    # Align unstructured frames
    test_df = test_df.sort_values(by=test_df.columns.to_list())
    test_df = test_df.reset_index(drop=True)

    # Retrive subject ids
    curr_subjects = list()
    subjects = deepcopy(subject_ids)

    # Test on existing directory
    for num_subjects in [1, 10, 10]:
        # Extend the subject id list
        curr_subjects.extend(random.sample(subjects, num_subjects - len(curr_subjects)))
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare generated results
        tests_io(
            f"-> Testing engineer-only with {len(curr_subjects)} subjects on existing directory.")
        engineer_and_compare(subject_ids=curr_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Test on empty directory
    engineered_dir = Path(TEMP_DIR, "engineered")
    for num_subjects in [11, 16]:
        if TEMP_DIR.is_dir():
            shutil.rmtree(engineered_dir)
        # Sample subject ids
        curr_subjects = random.sample(subjects, num_subjects)
        subjects = list(set(subjects) - set(curr_subjects))
        # Compare the processed data
        tests_io(f"-> Testing engineer-only with {len(curr_subjects)} subjects on empty directory.")
        engineer_and_compare(subject_ids=curr_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on empty directory.")
    # Test reducing subject count
    curr_subjects = random.sample(subjects, 1)
    engineer_and_compare(subject_ids=curr_subjects,
                         task_name=task_name,
                         extraction_style=extraction_style,
                         test_df=test_df)
    tests_io(f"-> Succeeded in reducing number of subjects.")


def extract_and_compare(subject_ids: list, test_subject_ids: list, extraction_style: str,
                        extracted_reader: ExtractedSetReader):
    return_entity: ExtractedSetReader = datasets.load_data(
        chunksize=(75835 if extraction_style == "iterative" else None),
        source_path=TEST_DATA_DEMO,
        subject_ids=subject_ids,
        storage_path=TEMP_DIR,
        extract=True)
    if extraction_style == "iterative":
        reader = return_entity
        # Some ids will not be processed due to no in unit subject events
        assert len(reader.subject_ids)
        missing_ids = list(set(subject_ids) - set(reader.subject_ids))
        # Make sure they are also not processed in the gt directory
        assert not set(missing_ids) & set(test_subject_ids)
        additional_ids = list(set(reader.subject_ids) - set(subject_ids))
        assert not additional_ids
        generated_dataset = reader.read_subjects(reader.subject_ids, read_ids=True)
    else:
        generated_dataset = return_entity
        assert len(generated_dataset)
        # Some ids will not be processed due to no in unit subject events
        missing_ids = list(set(subject_ids) - set(generated_dataset.keys()))
        # Make sure they are also not processed in the gt directory
        assert not set(missing_ids) & set(test_subject_ids)
        additional_ids = list(set(generated_dataset.keys()) - set(subject_ids))
        assert not additional_ids
    tests_io(f"-> The following IDs have not been extracted: {*missing_ids,}")
    test_dataset = extracted_reader.read_subjects(subject_ids, read_ids=True)
    compare_extracted_datasets(generated_dataset, test_dataset)


def process_and_compare(subject_ids: list, task_name: str, extraction_style: str,
                        test_data_dir: Path):
    # Process the dataset
    return_entity: ProcessedSetReader = datasets.load_data(
        chunksize=(75835 if extraction_style == "iterative" else None),
        source_path=TEST_DATA_DEMO,
        storage_path=TEMP_DIR,
        subject_ids=subject_ids,
        preprocess=True,
        task=task_name)

    listfile_df = pd.read_csv(Path(test_data_dir, "listfile.csv"))
    regex = r"(\d+)_episode(\d+)_timeseries\.csv"
    test_subject_ids = listfile_df.apply(lambda x: re.search(regex, x["stay"]).group(1), axis=1)
    test_subject_ids = test_subject_ids.astype(int)
    test_subject_ids = test_subject_ids.unique()

    if extraction_style == "iterative":
        reader = return_entity
        # Some ids will not be processed due to no in unit subject events
        missing_ids = list(set(subject_ids) - set(reader.subject_ids))
        # Make sure they are also not processed in the gt directory
        assert not set(missing_ids) & set(test_subject_ids)
        additional_ids = list(set(reader.subject_ids) - set(subject_ids))
        assert not additional_ids
        if reader.subject_ids:
            assert_reader_equals(reader, test_data_dir)

    else:
        generated_path = Path(TEMP_DIR, "processed", task_name)  # Outpath for task generation
        test_data_dir = Path(TEST_GT_DIR, "processed",
                             TASK_NAME_MAPPING[task_name])  # Ground truth data dir
        generated_samples = return_entity
        # Some ids will not be processed due to no in unit subject events
        missing_ids = list(set(subject_ids) - set(generated_samples["X"].keys()))
        # Make sure they are also not processed in the gt directory
        assert not set(missing_ids) & set(test_subject_ids)
        additional_ids = list(set(generated_samples["X"].keys()) - set(subject_ids))
        assert not additional_ids
        X = generated_samples["X"]
        y = generated_samples["y"]
        if len(X):
            assert_dataset_equals(X, y, generated_path, test_data_dir)


def engineer_and_compare(subject_ids: list, task_name: str, extraction_style: str, test_df: Path):
    return_entity: ProcessedSetReader = datasets.load_data(
        chunksize=(75835 if extraction_style == "iterative" else None),
        source_path=TEST_DATA_DEMO,
        storage_path=TEMP_DIR,
        subject_ids=subject_ids,
        engineer=True,
        task=task_name)
    if extraction_style == "iterative":
        reader = return_entity
        generated_samples = reader.read_samples(read_ids=True)
        # Some ids will not be processed due to no in unit subject events
        missing_ids = list(set(subject_ids) - set(generated_samples["X"].keys()))
        # Make sure they are also not processed in the gt directory
        assert not set(missing_ids) & set(test_df["SUBJECT_ID"].unique())
        additional_ids = list(set(generated_samples["X"].keys()) - set(subject_ids))
        assert not additional_ids
    else:
        generated_samples = return_entity
        # Some ids will not be processed due to no in unit subject events
        missing_ids = list(set(subject_ids) - set(generated_samples["X"].keys()))
        # Make sure they are also not processed in the gt directory
        assert not set(missing_ids) & set(test_df["SUBJECT_ID"].unique())
        additional_ids = list(set(generated_samples["X"].keys()) - set(subject_ids))
        assert not additional_ids
    generated_df = concatenate_dataset(generated_samples["X"])

    # Align unstructured frames
    if generated_df is None:
        tests_io(f"-> No data generated for {task_name} task.")
        return
    columns = generated_df.columns.to_list()

    generated_samples = return_entity

    generated_df = generated_df.sort_values(by=columns)
    generated_df = generated_df.reset_index(drop=True)
    stay_ids = generated_df["ICUSTAY_ID"].unique()

    curr_test_df = test_df[test_df["ICUSTAY_ID"].isin(stay_ids.astype("str").tolist())]
    curr_test_df = curr_test_df.sort_values(by=columns)
    curr_test_df = curr_test_df.reset_index(drop=True)

    assert_dataframe_equals(generated_df,
                            curr_test_df,
                            rename={"hours": "Hours"},
                            normalize_by="groundtruth")


if __name__ == "__main__":
    extraction_reader = datasets.load_data(chunksize=75835,
                                           source_path=TEST_DATA_DEMO,
                                           storage_path=SEMITEMP_DIR)
    icu_history = extraction_reader.read_csv("icu_history.csv")
    subjects = icu_history["SUBJECT_ID"].astype(int).unique().tolist()
    for extraction_style in ["iterative"]:  #"compact", "iterative"]:
        if TEMP_DIR.is_dir():
            shutil.rmtree(str(TEMP_DIR))
        test_subject_ids_extraction(extraction_reader, subjects, extraction_style)
        for task in TASK_NAMES:
            if not Path(SEMITEMP_DIR, "processed", task).is_dir():
                reader = datasets.load_data(chunksize=75835,
                                            source_path=TEST_DATA_DEMO,
                                            storage_path=SEMITEMP_DIR,
                                            preprocess=True,
                                            task=task)

            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_subject_ids_preprocessing_only(task, subjects, extraction_style)
            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_subject_ids_engineer_only(task, subjects, extraction_style)
            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_subject_ids_preprocessing(task, subjects, extraction_style)
            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_subject_ids_engineer(task, subjects, extraction_style)
