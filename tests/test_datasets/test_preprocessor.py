import datasets
import pandas as pd
import re
import pytest
from pathlib import Path
from datasets.readers import ExtractedSetReader
from utils.IO import *
from tests.pytest_utils import copy_dataset
from tests.decorators import repeat
from tests.tsettings import *
from settings import *
from tests.pytest_utils.preprocessing import assert_reader_equals, assert_dataset_equals

kwargs = {
    "IHM": {
        "minimum_length_of_stay": IHM_SETTINGS['label_start_time']
    },
    "DECOMP": {
        "label_start_time": DECOMP_SETTINGS['label_start_time']
    },
    "LOS": {
        "label_start_time": LOS_SETTINGS['label_start_time']
    },
    "PHENO": {},
    "MULTI": {
        "file_suffix": "h5",
        "test_index": "filename"
    }
}


@pytest.mark.parametrize("task_name", TASK_NAMES)
@repeat(2)
def test_iterative_processing_task(task_name: str):
    """_summary_

    Args:
        task_name (str): _description_
    """
    tests_io(f"Test iterative preprocessor for task {task_name}", level=0)

    # Resolve task name
    generated_path = Path(TEMP_DIR, "processed", task_name)  # Outpath for task generation
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    copy_dataset("extracted")

    # Load
    reader = datasets.load_data(chunksize=75835,
                                source_path=TEST_DATA_DEMO,
                                storage_path=TEMP_DIR,
                                preprocess=True,
                                task=generated_path.name)

    assert_file_creation(reader.root_path, test_data_dir, **kwargs[generated_path.name])

    tests_io(f"All {task_name} files have been created as expected")
    # Compare the dataframes in the directory
    assert_reader_equals(reader, test_data_dir, **kwargs[generated_path.name])
    tests_io(f"{task_name} preprocessing successfully tested against original code!")

    return


@pytest.mark.parametrize("task_name", TASK_NAMES)
@repeat(2)
def test_compact_processing_task(task_name: str):
    """_summary_

    Args:
        task_name (str): _description_
    """
    tests_io(f"Test compact preprocessor for task {task_name}", level=0)

    # Resolve task name
    generated_path = Path(TEMP_DIR, "processed", task_name)  # Outpath for task generation
    test_data_dir = Path(TEST_GT_DIR, "processed",
                         TASK_NAME_MAPPING[task_name])  # Ground truth data dir

    copy_dataset("extracted")

    # Load/Create data
    dataset = datasets.load_data(source_path=TEST_DATA_DEMO,
                                 storage_path=TEMP_DIR,
                                 preprocess=True,
                                 task=generated_path.name)

    assert_file_creation(generated_path, test_data_dir, **kwargs[generated_path.name])

    tests_io(f"All {task_name} files have been created as expected")
    # Compare the dataframes in the directory
    assert_dataset_equals(dataset["X"], dataset["y"], generated_path, test_data_dir,
                          **kwargs[generated_path.name])
    tests_io(f"{task_name} preprocessing successfully tested against original code!")

    return


def assert_file_creation(root_path: Path,
                         test_data_dir: Path,
                         label_start_time: float = None,
                         minimum_length_of_stay: float = None,
                         file_suffix: str = None,
                         **kwargs):
    """_summary_

    Args:
        reader (SampleSetReader): _description_
        test_data_dir (Path): _description_
    """
    count = 0
    removed_count = 0
    subject_ids_checked = list()
    tests_io(f"Total stays checked: {count}\n"
             f"Total subjects checked: {len(set(subject_ids_checked))}\n"
             f"Subjects removed correctly: {removed_count}")

    assert root_path.is_dir(), f"Generated directory {root_path} does not exist"
    assert test_data_dir.is_dir(), f"Test directory {test_data_dir} does not exist"

    file_suffix = "csv" if file_suffix is None else file_suffix
    # Test wether all files have been created correctly
    for file_path in Path(test_data_dir).iterdir():
        if file_path.name == "listfile.csv":
            continue
        match = re.search(r"(\d+)_episode(\d+)_timeseries\.csv", file_path.name)
        subject_id = int(match.group(1))
        stay_id = int(match.group(2))
        # Files that are not longer than the minimum that needs to be elapsed before label creation are removed
        test_data = pd.read_csv(Path(test_data_dir,
                                     f"{subject_id}_episode{stay_id}_timeseries.csv"),
                                na_values=[''],
                                keep_default_na=False)
        # Increment for logging
        count += 1
        subject_ids_checked.append(subject_id)

        if label_start_time is not None:
            if label_start_time > test_data["Hours"].max():
                assert not Path(root_path, str(subject_id), f"X_{stay_id}.{file_suffix}").is_file(
                ), f"Sample file X_{stay_id}.csv for subject {subject_id} should be deleted due to minimum label start time."
                assert not Path(root_path, str(subject_id), f"y_{stay_id}.{file_suffix}").is_file(
                ), f"Label file y_{stay_id}.csv for subject {subject_id} should be deleted due to minimum label start time."
                removed_count += 1
                continue

        if minimum_length_of_stay is not None:
            test_episode_data = pd.read_csv(Path(test_data_dir.parent.parent, "extracted",
                                                 str(subject_id),
                                                 f"episode{stay_id}.{file_suffix}"),
                                            na_values=[''],
                                            keep_default_na=False)
            if test_episode_data["Length of Stay"][0] > minimum_length_of_stay:
                assert not Path(root_path, str(subject_id), f"X_{stay_id}.{file_suffix}").is_file(
                ), f"Sample file X_{stay_id}.csv for subject {subject_id} should be deleted due to minimum length of stay."
                assert not Path(root_path, str(subject_id), f"y_{stay_id}.{file_suffix}").is_file(
                ), f"Label file y_{stay_id}.csv for subject {subject_id} should be deleted due to minimum length of stay."
                removed_count += 1
                continue

        assert Path(root_path, str(subject_id), f"X_{stay_id}.{file_suffix}").is_file(
        ), f"Missing sample file X_{stay_id}.csv for subject {subject_id}"
        assert Path(root_path, str(subject_id), f"y_{stay_id}.{file_suffix}").is_file(
        ), f"Missing label file y_{stay_id}.csv for subject {subject_id}"
        tests_io(
            f"Total stays checked: {count}\n"
            f"Total subjects checked: {len(set(subject_ids_checked))}\n"
            f"Subjects removed correctly: {removed_count}",
            flush_block=True)


if __name__ == "__main__":
    import shutil
    _ = datasets.load_data(chunksize=75835, source_path=TEST_DATA_DEMO, storage_path=SEMITEMP_DIR)
    for task in ["MULTI"]:  #TASK_NAMES:
        if Path(TEMP_DIR).is_dir():
            shutil.rmtree(str(Path(TEMP_DIR)))
        test_compact_processing_task(task)
        if Path(TEMP_DIR).is_dir():
            shutil.rmtree(str(TEMP_DIR))
        test_iterative_processing_task(task)
