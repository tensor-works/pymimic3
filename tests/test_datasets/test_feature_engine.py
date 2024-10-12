import datasets
import pytest
import pandas as pd
from pathlib import Path
from utils.IO import *
from tests.tsettings import *
from tests.pytest_utils import copy_dataset
from tests.pytest_utils.general import assert_dataframe_equals
from tests.pytest_utils.decorators import repeat
from tests.pytest_utils.feature_engineering import extract_test_ids, concatenate_dataset

compare_mode_map = {
    "IHM": "single_entry",
    "LOS": "multiline",
    "PHENO": "single_entry",
    "DECOMP": "multiline"
}


@pytest.mark.parametrize("task_name", TASK_NAMES)
@repeat(2)
def test_iterative_engineer_task(task_name: str):
    """ Tests the feature engineering for in hospital mortality, done for linear linear models.
    Preprocessing tests need to be completed before running this test.
    """

    tests_io(f"Test case iterative engineering for task {task_name}", level=0)

    generated_path = Path(TEMP_DIR, "engineered", task_name)  # Outpath for task genergeneraation
    if task_name == "MULTI":
        with pytest.raises(ValueError) as e:
            reader = datasets.load_data(chunksize=75837,
                                        source_path=TEST_DATA_DEMO,
                                        storage_path=TEMP_DIR,
                                        engineer=True,
                                        task=generated_path.name)
            assert e.value == "Task 'MULTI' is not supported for feature engineering."
        tests_io(f"Succesfully tested task fail for {task_name}.")
        return
    test_data_dir = Path(TEST_GT_DIR, "engineered",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))

    tests_io(f"Engineering data for task {task_name}.")
    reader = datasets.load_data(chunksize=75837,
                                source_path=TEST_DATA_DEMO,
                                storage_path=TEMP_DIR,
                                engineer=True,
                                task=generated_path.name)

    # Concatenate engineered samples into labeleod data frame
    generated_samples = reader.read_samples(read_ids=True)
    generated_df = concatenate_dataset(generated_samples["X"])

    # Load test data and extract ids
    test_df = pd.read_csv(Path(test_data_dir, "X.csv"), low_memory=False)
    test_df = extract_test_ids(test_df)

    # Align unstructured frames
    columns = list(reversed(generated_df.columns.to_list()))
    generated_df = generated_df[columns]
    generated_df = generated_df.round(10)
    generated_df = generated_df.sort_values(by=generated_df.columns.to_list())
    generated_df = generated_df.reset_index(drop=True)

    test_df = test_df[columns]
    test_df = test_df.astype(generated_df.dtypes)
    test_df = test_df.round(10)
    test_df = test_df.sort_values(by=columns)
    test_df = test_df.reset_index(drop=True)

    assert_dataframe_equals(generated_df,
                            test_df,
                            rename={"hours": "Hours"},
                            normalize_by="groundtruth")

    tests_io("Testing against ground truth data.")
    tests_io(f"Total stays checked: {generated_df['SUBJECT_ID'].nunique()}\n"
             f"Total subjects checked: {generated_df['ICUSTAY_ID'].nunique()}\n"
             f"Total samples checked: {len(generated_df)}")

    tests_io(f"{task_name} feature engineering successfully tested against original code!")

    return


@pytest.mark.parametrize("task_name", TASK_NAMES)
@repeat(2)
def test_compact_engineer_task(task_name: str):
    """ Tests the feature engineering for in hospital mortality, done for linear linear models.
    Preprocessing tests need to be completed before running this test.
    """

    tests_io(f"Test case compact engineering for task {task_name}", level=0)
    task_name_mapping = dict(zip(TASK_NAMES, FTASK_NAMES))

    generated_path = Path(TEMP_DIR, "engineered", task_name)  # Outpath for task generation
    if task_name == "MULTI":
        with pytest.raises(ValueError) as e:
            dataset = datasets.load_data(source_path=TEST_DATA_DEMO,
                                         storage_path=TEMP_DIR,
                                         engineer=True,
                                         task=generated_path.name)
            assert e.value == "Task 'MULTI' is not supported for feature engineering."
        tests_io(f"Succesfully tested task fail for {task_name}.")
        return
    test_data_dir = Path(TEST_GT_DIR, "engineered",
                         task_name_mapping[task_name])  # Ground truth data dir

    copy_dataset("extracted")
    copy_dataset(Path("processed", task_name))

    # Preprocess the data
    tests_io(f"Engineering data for task {task_name}.")
    dataset = datasets.load_data(source_path=TEST_DATA_DEMO,
                                 storage_path=TEMP_DIR,
                                 engineer=True,
                                 task=generated_path.name)

    # Concatenate engineered samples into labeled data frame
    generated_df = concatenate_dataset(dataset["X"])

    # Load test data and extract ids
    test_df = pd.read_csv(Path(test_data_dir, "X.csv"))
    test_df = extract_test_ids(test_df)

    # Align unstructured frames
    generated_df = generated_df.round(10)
    generated_df = generated_df.sort_values(by=generated_df.columns.to_list())
    generated_df = generated_df.reset_index(drop=True)

    test_df = test_df.astype(generated_df.dtypes)
    test_df = test_df.round(10)
    test_df = test_df.sort_values(by=generated_df.columns.to_list())
    test_df = test_df.reset_index(drop=True)

    assert_dataframe_equals(generated_df,
                            test_df,
                            rename={"hours": "Hours"},
                            normalize_by="groundtruth")
    tests_io("Testing against ground truth data.")
    tests_io(f"Total stays checked: {generated_df['SUBJECT_ID'].nunique()}\n"
             f"Total subjects checked: {generated_df['ICUSTAY_ID'].nunique()}\n"
             f"Total samples checked: {len(generated_df)}")

    tests_io(f"{task_name} feature engineering successfully tested against original code!")

    return


if __name__ == "__main__":
    import shutil
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    for task in TASK_NAMES:
        if task == "MULTI":
            continue
        _ = datasets.load_data(chunksize=75835,
                               task=task,
                               preprocess=True,
                               source_path=TEST_DATA_DEMO,
                               storage_path=SEMITEMP_DIR)
        if TEMP_DIR.is_dir():
            shutil.rmtree(str(TEMP_DIR))
        test_compact_engineer_task(task)
        if TEMP_DIR.is_dir():
            shutil.rmtree(str(TEMP_DIR))
        test_iterative_engineer_task(task)
