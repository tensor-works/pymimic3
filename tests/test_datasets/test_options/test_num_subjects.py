import datasets
import pandas as pd
import shutil
import pytest
from tests.pytest_utils import copy_dataset
from typing import Dict
from pathlib import Path
from utils.IO import *
from tests.settings import *
from tests.pytest_utils.general import assert_dataframe_equals
from tests.pytest_utils.preprocessing import assert_reader_equals, assert_dataset_equals
from tests.pytest_utils.feature_engineering import extract_test_ids, concatenate_dataset
from tests.pytest_utils.extraction import compare_extracted_datasets
from datasets.readers import ExtractedSetReader, ProcessedSetReader


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
def test_num_subjects_extraction(extracted_reader: ExtractedSetReader, extraction_style: str):
    # Test only extraction
    tests_io(f"Test case num subjects for extraction", level=0)

    # Test on existing directory
    for num_subjects in [1, 11, 21]:
        tests_io(f"-> Testing extract-only with {num_subjects} subjects on existing directory.")
        extract_and_compare(num_subjects=num_subjects,
                            extraction_style=extraction_style,
                            extracted_reader=extracted_reader)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Test on empty directory
    extracted_dir = Path(TEMP_DIR, "extracted")
    for num_subjects in [11, 21]:
        # Remove existing dir
        if extracted_dir.is_dir():
            shutil.rmtree(str(extracted_dir))
        # Compare extraction
        tests_io(f"-> Testing extract-only with {num_subjects} subjects on empty directory.")
        extract_and_compare(num_subjects=num_subjects,
                            extraction_style=extraction_style,
                            extracted_reader=extracted_reader)

    tests_io(f"-> Succeeded in testing on empty directory.")

    # Test reducing subject count
    extract_and_compare(num_subjects=1,
                        extraction_style=extraction_style,
                        extracted_reader=extracted_reader)
    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_num_subjects_preprocessing_only(task_name: str, extraction_style: str):
    tests_io(f"Test case num subjects for preprocessing-only for task {task_name}.", level=0)
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Make sure extracted data is available
    tests_io(f"Preparing fully extracted directory")
    copy_dataset("extracted")

    # Test on existing directory
    for num_subjects in [1, 11, 21]:
        tests_io(f"-> Testing preprocessing-only with {num_subjects}"
                 " subjects on existing directory.")
        process_and_compare(num_subjects=num_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Test on empty directory
    processed_dir = Path(TEMP_DIR, "processed")
    for num_subjects in [11, 21]:
        if processed_dir.is_dir():
            shutil.rmtree(str(processed_dir))
        tests_io(f"-> Testing preprocessing-only with {num_subjects}"
                 " subjects on empty directory.")
        process_and_compare(num_subjects=num_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on empty directory.")

    # Test reducing subject count
    process_and_compare(num_subjects=1,
                        task_name=task_name,
                        extraction_style=extraction_style,
                        test_data_dir=test_data_dir)
    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_num_subjects_engineer_only(task_name: str, extraction_style: str):
    tests_io(f"Test case num subjects for engineering-only for task {task_name}.", level=0)
    # Test only engineering
    tests_io(f"Preparing fully preprocessed directory")
    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))

    test_data_dir = Path(TEST_GT_DIR, "engineered",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Load test data and extract ids
    test_df = pd.read_csv(Path(test_data_dir, "X.csv"), na_values=[''], keep_default_na=False)
    test_df = extract_test_ids(test_df)

    # Align unstructured frames
    test_df = test_df.sort_values(by=test_df.columns.to_list())
    test_df = test_df.reset_index(drop=True)

    # Test on existing directory
    for num_subjects in [1, 11, 16]:
        tests_io(f"-> Testing engineer-only with {num_subjects} subjects on existing directory.")
        engineer_and_compare(num_subjects=num_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Test on empty directory
    engineered_dir = Path(TEMP_DIR, "engineered")
    for num_subjects in [11, 16]:
        if TEMP_DIR.is_dir():
            shutil.rmtree(engineered_dir)
        tests_io(f"-> Testing engineer-only with {num_subjects} subjects on empty directory.")
        engineer_and_compare(num_subjects=num_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on empty directory.")

    # Test reducing subject count
    engineer_and_compare(num_subjects=1,
                         task_name=task_name,
                         extraction_style=extraction_style,
                         test_df=test_df)
    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_num_subjects_process(task_name: str, extraction_style: str):
    tests_io(f"Test case num subjects for preprocessing from scratch for task {task_name}.",
             level=0)
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Test on existing directory
    for num_subjects in [1, 6, 11, 16, 21]:
        tests_io(
            f"-> Testing preprocessing {num_subjects} subjects on existing directory from scratch.")
        process_and_compare(num_subjects=num_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Test on empty directory
    for num_subjects in [6, 11, 16, 21]:
        if TEMP_DIR.is_dir():
            shutil.rmtree(str(Path(TEMP_DIR)))
        tests_io(
            f"-> Testing preprocessing {num_subjects} subjects on empty director from scratch.")
        process_and_compare(num_subjects=num_subjects,
                            task_name=task_name,
                            extraction_style=extraction_style,
                            test_data_dir=test_data_dir)

    tests_io(f"-> Succeeded in testing on empty directory.")
    # Test reducing subject count
    process_and_compare(num_subjects=1,
                        task_name=task_name,
                        extraction_style=extraction_style,
                        test_data_dir=test_data_dir)
    tests_io(f"-> Succeeded in reducing number of subjects.")


@pytest.mark.parametrize("extraction_style", ["iterative", "compact"])
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_num_subjects_engineer(task_name: str, extraction_style: str):
    tests_io(f"Test case num subjects for engineering from scratch for task {task_name}.", level=0)

    test_data_dir = Path(TEST_GT_DIR, "engineered",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    # Load test data and extract ids
    test_df = pd.read_csv(Path(test_data_dir, "X.csv"), na_values=[''], keep_default_na=False)
    test_df = extract_test_ids(test_df)

    # Align unstructured frames
    test_df = test_df.sort_values(by=test_df.columns.to_list())
    test_df = test_df.reset_index(drop=True)

    # Test on existing directory
    for num_subjects in [1, 11, 16]:
        tests_io(
            f"-> Testing engineering {num_subjects} subjects on existing directory from scratch.")
        engineer_and_compare(num_subjects=num_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on existing directory.")

    # Test on empty directory
    for num_subjects in [11, 16]:
        if TEMP_DIR.is_dir():
            shutil.rmtree(TEMP_DIR)
        tests_io(
            f"-> Testing preprocessing {num_subjects} subjects on empty directory from scratch.")
        engineer_and_compare(num_subjects=num_subjects,
                             task_name=task_name,
                             extraction_style=extraction_style,
                             test_df=test_df)

    tests_io(f"-> Succeeded in testing on empty directory.")

    # Test reducing subject count
    engineer_and_compare(num_subjects=1,
                         task_name=task_name,
                         extraction_style=extraction_style,
                         test_df=test_df)
    tests_io(f"-> Succeeded in reducing number of subjects.")


def extract_and_compare(num_subjects: int, extraction_style: str,
                        extracted_reader: ExtractedSetReader):
    return_entity = datasets.load_data(
        chunksize=(75835 if extraction_style == "iterative" else None),
        source_path=TEST_DATA_DEMO,
        num_subjects=num_subjects,
        storage_path=TEMP_DIR,
        extract=True)
    if extraction_style == "iterative":
        reader: ExtractedSetReader = return_entity
        assert len(reader.subject_ids) == num_subjects
        subject_ids = reader.subject_ids
        generated_dataset = reader.read_subjects(reader.subject_ids, read_ids=True)
    else:
        generated_dataset: dict = return_entity
        assert len(generated_dataset) == num_subjects
        subject_ids = list(generated_dataset.keys())
    test_dataset = extracted_reader.read_subjects(subject_ids, read_ids=True)

    compare_extracted_datasets(generated_dataset, test_dataset)


def process_and_compare(num_subjects: int, task_name: str, extraction_style: str,
                        test_data_dir: Path):
    return_entity: ProcessedSetReader = datasets.load_data(
        chunksize=(75835 if extraction_style == "iterative" else None),
        source_path=TEST_DATA_DEMO,
        storage_path=TEMP_DIR,
        num_subjects=num_subjects,
        preprocess=True,
        task=task_name)

    if extraction_style == "iterative":
        reader = return_entity
        assert len(reader.subject_ids) == num_subjects
        assert_reader_equals(reader, test_data_dir)
    else:
        generated_path = Path(TEMP_DIR, "processed", task_name)  # Outpath for task generation
        test_data_dir = Path(TEST_GT_DIR, "processed",
                             TASK_NAME_MAPPING[task_name])  # Ground truth data dir
        generated_samples = return_entity
        assert len(generated_samples["X"]) == num_subjects
        X = generated_samples["X"]
        y = generated_samples["y"]
        assert_dataset_equals(X, y, generated_path, test_data_dir)


def engineer_and_compare(num_subjects: int, task_name: str, extraction_style: str, test_df: Path):
    return_entity: ProcessedSetReader = datasets.load_data(
        chunksize=(75835 if extraction_style == "iterative" else None),
        source_path=TEST_DATA_DEMO,
        storage_path=TEMP_DIR,
        num_subjects=num_subjects,
        engineer=True,
        task=task_name)
    if extraction_style == "iterative":
        reader = return_entity
        generated_samples = reader.read_samples(read_ids=True)
    else:
        generated_samples = return_entity
    assert len(generated_samples["X"])  # Generated df is None if this doesn't pass
    generated_df = concatenate_dataset(generated_samples["X"])
    # Align unstructured frames
    generated_df = generated_df.sort_values(by=generated_df.columns.to_list())
    generated_df = generated_df.reset_index(drop=True)
    stay_ids = generated_df["ICUSTAY_ID"].unique()

    curr_test_df = test_df[test_df["ICUSTAY_ID"].isin(stay_ids.astype("str").tolist())]
    curr_test_df = curr_test_df.reset_index(drop=True)

    assert generated_df["SUBJECT_ID"].nunique() == num_subjects
    assert_dataframe_equals(generated_df,
                            curr_test_df,
                            rename={"hours": "Hours"},
                            normalize_by="groundtruth")
    tests_io("Engineered subjects are identical!")


if __name__ == "__main__":
    extraction_reader = datasets.load_data(chunksize=75835,
                                           source_path=TEST_DATA_DEMO,
                                           storage_path=SEMITEMP_DIR)
    for extraction_style in ["compact", "iterative"]:
        if TEMP_DIR.is_dir():
            shutil.rmtree(str(TEMP_DIR))
        test_num_subjects_extraction(extraction_reader, extraction_style)
        for task in TASK_NAMES:
            if not Path(SEMITEMP_DIR, "processed", task).is_dir():
                reader = datasets.load_data(chunksize=75835,
                                            source_path=TEST_DATA_DEMO,
                                            storage_path=SEMITEMP_DIR,
                                            preprocess=True,
                                            task=task)
            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_num_subjects_preprocessing_only(task, extraction_style)
            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_num_subjects_engineer_only(task, extraction_style)
            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_num_subjects_process(task, extraction_style)
            if TEMP_DIR.is_dir():
                shutil.rmtree(str(TEMP_DIR))
            test_num_subjects_engineer(task, extraction_style)

    tests_io("All tests completed successfully!")
