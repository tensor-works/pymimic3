import datasets
import shutil
import pandas as pd
from tests.decorators import repeat
from datasets.readers import ExtractedSetReader
from pathlib import Path
from utils.IO import *
from tests.settings import *
from tests.pytest_utils.general import assert_dataframe_equals

top_level_files = ["diagnoses.csv", "icu_history.csv"]


@repeat(2)
def test_iterative_extraction():
    """ Tests the preprocessing handled by the load data method of the dataset module (extracted data).
    """
    tests_io("Test case iterative extraction.", level=0)
    # Extract the data
    test_data_dir = Path(TEST_GT_DIR, "extracted")
    reader: ExtractedSetReader = datasets.load_data(
        chunksize=900000,  # Using a large number to run on single read for comparision
        source_path=TEST_DATA_DEMO,
        storage_path=TEMP_DIR)

    compare_diagnoses_and_history(test_data_dir)
    tests_io("Datset creation successfully tested against original code!")
    compare_subject_directories(test_data_dir, reader.read_subjects(read_ids=True))
    tests_io("Dataset restoration successfully tested against original code!")


@repeat(2)
def test_compact_extraction():
    # Extract the data
    tests_io("Test case compact extraction.", level=0)
    test_data_dir = Path(TEST_GT_DIR, "extracted")
    dataset = datasets.load_data(source_path=TEST_DATA_DEMO, storage_path=TEMP_DIR)
    compare_diagnoses_and_history(test_data_dir)
    tests_io("Dataset creation successfully tested against original code!")
    compare_subject_directories(test_data_dir, dataset)
    tests_io("Dataset restoration successfully tested against original code!")


def compare_diagnoses_and_history(test_data_dir: Path):
    # Compare files to ground truth
    for file_name in top_level_files:
        file_settings = TEST_SETTINGS[file_name]
        test_file_name = file_settings["name_mapping"]
        tests_io(f"Test {file_name}")
        generated_df = pd.read_csv(Path(TEMP_DIR, "extracted", file_name))
        test_df = pd.read_csv(Path(test_data_dir, test_file_name))
        if "columns" in file_settings:
            generated_df = generated_df[file_settings["columns"]]
        assert_dataframe_equals(generated_df, test_df, normalize_by="groundtruth")


def compare_subject_directories(test_data_dir: Path, dataset: dict):
    generated_dirs = [
        directory for directory in Path(TEMP_DIR, "extracted").iterdir()
        if directory.is_dir() and directory.name.isnumeric()
    ]
    assert len(dataset)

    tests_io("subject_events.csv: 0 subject\n"
             "subject_diagnoses.csv: 0 subject\n"
             "timeseries.csv: 0 subject, 0 stays\n"
             "episodic_data.csv: 0 subject")

    counts = {
        "subject_events.csv": 0,
        "subject_diagnoses.csv": 0,
        "timeseries_stay_id.csv": 0,
        "episodic_data.csv": 0
    }

    ts_counts = 0

    # Compare the subject directories
    for directory in generated_dirs:
        stay_ids = list()

        # File equivalences:
        # Generated: Ground truth
        # subject_events.csv: events.csv
        # subject_diagnoses.csv: diagnoses.csv
        # timeseries_stay_id.csv': episodestay_id_timeseries.csv
        # episodic_data.csv: episodestay_id.csv

        for file_name, file_settings in TEST_SETTINGS.items():
            if file_name in top_level_files:  # already checked
                continue
            test_file_name = file_settings["name_mapping"]
            # Stay ID from early files
            if "stay_id" in file_name or "stay_id" in test_file_name:
                # Episodic data and timeseries are stored per stay_id in ground truth
                for stay_id in stay_ids:
                    # Insert the stay id
                    stay_file_name = file_name.replace("stay_id", str(int(stay_id)))
                    test_stay_file_name = test_file_name.replace("stay_id", str(int(stay_id)))
                    # Read the generated df
                    if stay_file_name == "episodic_data.csv":
                        continue
                        # Incorrect in original repository. Remove continue if issue is resolved
                        # https://github.com/YerevaNN/mimic3-benchmarks/issues/101
                        # Episodic data stored together in this implementation
                        generated_df = generated_df[generated_df["Icustay"] == stay_id].reset_index(
                            drop=True)

                    generated_df = dataset[int(directory.name)][str(Path(file_name).stem).replace(
                        "_stay_id", "")][stay_id].reset_index()

                    # Read the test df
                    test_df = pd.read_csv(Path(test_data_dir, directory.name, test_stay_file_name))
                    assert_dataframe_equals(generated_df, test_df, rename=file_settings["rename"])

                if file_name == "timeseries_stay_id.csv":
                    counts["timeseries_stay_id.csv"] += len(stay_ids)
                    ts_counts += 1
                elif file_name == "episodic_data.csv":
                    counts[
                        "episodic_data.csv"] += 0  # Incorrect in original repository. Remove 0 if issue is resolved
                tests_io(
                    f"subject_events.csv: {counts['subject_events.csv']} subjects\n"
                    f"subject_diagnoses.csv: {counts['subject_diagnoses.csv']} subjects\n"
                    f"timeseries.csv: {ts_counts} subjects, {counts['timeseries_stay_id.csv']} stays\n"
                    f"episodic_data.csv: {counts[file_name]} subjects",
                    flush_block=True)
            else:
                generated_df = dataset[int(directory.name)][str(Path(file_name).stem)]
                if file_name == "subject_events.csv":
                    stay_ids = generated_df["ICUSTAY_ID"].unique()

                test_df = pd.read_csv(Path(test_data_dir, directory.name, test_file_name))
                assert_dataframe_equals(generated_df, test_df, rename=file_settings["rename"])
                counts[file_name] += 1
                tests_io(
                    f"subject_events.csv: {counts['subject_events.csv']} subjects\n"
                    f"subject_diagnoses.csv: {counts['subject_diagnoses.csv']} subjects\n"
                    f"timeseries.csv: {ts_counts} subjects, {counts['timeseries_stay_id.csv']} stays\n"
                    f"episodic_data.csv: {counts[file_name]} subjects",
                    flush_block=True)


if __name__ == "__main__":

    if TEMP_DIR.is_dir():
        shutil.rmtree(TEMP_DIR)
    test_iterative_extraction()
    if TEMP_DIR.is_dir():
        shutil.rmtree(TEMP_DIR)
    test_compact_extraction()
    if TEMP_DIR.is_dir():
        shutil.rmtree(TEMP_DIR)
