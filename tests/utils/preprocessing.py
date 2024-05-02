import shelve
import pandas as pd
from pathlib import Path
from utils.IO import *
from tests.utils.general import assert_dataframe_equals
from datasets.readers import ProcessedSetReader
from tests.settings import *


def assert_reader_equals(reader: ProcessedSetReader, test_data_dir: Path):
    """_summary_

    Args:
        reader (SampleSetReader): _description_
        test_data_dir (Path): _description_
    """
    assert reader.subject_ids, "The reader subjects are empty! Extraction must have failed."
    subject_count = 0
    stay_count = 0
    tests_io(f"Stays frames compared: {stay_count}\n" f"Total subjects checked: {subject_count}")
    for subject_id in reader.subject_ids:
        X_stays, y_stays = reader.read_sample(subject_id, read_ids=True).values()
        subject_count += 1
        stay_count += assert_subject_data_equals(subject_id, X_stays, y_stays, test_data_dir,
                                                 reader.root_path)
        tests_io(f"Compared subjects: {subject_count}\n"
                 f"Compared stays: {stay_count}\n",
                 flush_block=True)


def assert_dataset_equals(X: dict, y: dict, generated_dir: Path, test_data_dir: Path):
    """_summary_

    Args:
        reader (SampleSetReader): _description_
        test_data_dir (Path): _description_
    """
    assert len(X) and len(y), "The reader subjects are empty! Extraction must have failed."
    subject_count = 0
    stay_count = 0
    tests_io(f"Stays frames compared: {stay_count}\n" f"Total subjects checked: {subject_count}")
    for subject_id in X:
        X_stays, y_stays = X[subject_id], y[subject_id]
        subject_count += 1
        stay_count += assert_subject_data_equals(subject_id, X_stays, y_stays, test_data_dir,
                                                 generated_dir)
        tests_io(f"Compared subjects: {subject_count}\n"
                 f"Compared stays: {stay_count}\n",
                 flush_block=True)


def assert_subject_data_equals(subject_id: int, X_stays: dict, y_stays: dict, test_data_dir: Path,
                               root_path: Path):
    stay_count = 0
    for stay_id in X_stays.keys():
        X = X_stays[stay_id]
        y = y_stays[stay_id]
        try:
            test_data = pd.read_csv(
                Path(test_data_dir, f"{subject_id}_episode{stay_id}_timeseries.csv"))
        except:
            raise FileNotFoundError(f"Test set is missing {subject_id}"
                                    "_episode{stay_id}_timeseries.csv")

        # Testing the preprocessing tracker at the same time
        with shelve.open(str(Path(root_path, "progress"))) as db:
            assert int(subject_id) in db["subjects"]
            assert int(stay_id) in db["subjects"][subject_id]
            assert db["subjects"][subject_id][stay_id] == len(y)
            assert db["subjects"][subject_id]["total"] == sum([
                lengths for stay_id, lengths in db["subjects"][subject_id].items()
                if stay_id != "total"
            ])

        assert_dataframe_equals(X.reset_index(),
                                test_data, {"hours": "Hours"},
                                normalize_by="groundtruth")
        stay_count += 1
    return stay_count
