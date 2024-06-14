import datasets
import os
import pandas as pd
import re
import shutil
import random
import pytest
from pathlib import Path
from utils.IO import *
from tests.tsettings import *
from datasets.readers import ExtractedSetReader
from datasets.mimic_utils import convert_dtype_dict

ground_truth_subject_ids = [
    int(subject_dir.name)
    for subject_dir in Path(TEST_GT_DIR, "extracted").iterdir()
    if subject_dir.name.isnumeric()
]

ground_truth_stay_ids = [
    int(icustay) for subject_dir in Path(TEST_GT_DIR, "extracted").iterdir()
    if subject_dir.name.isnumeric()
    for icustay in pd.read_csv(Path(subject_dir, "stays.csv"),
                               na_values=[''],
                               keep_default_na=False).ICUSTAY_ID.to_numpy().tolist()
]

# Consider storing this somewhere else
frame_properties = {
    "timeseries": {
        "columns": [
            'Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen',
            'Glascow coma scale eye opening', 'Glascow coma scale motor response',
            'Glascow coma scale total', 'Glascow coma scale verbal response', 'Glucose',
            'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'pH',
            'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight'
        ],
        "index": "hours"
    },
    "episodic_data": {
        "columns": ['AGE', 'LOS', 'MORTALITY', 'GENDER', 'ETHNICITY', 'Height', 'Weight'],
        "index": "Icustay"
    },
    "subject_events": {
        "columns": [
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'
        ],
        "index": None
    },
    "subject_diagnoses": {
        "columns": [
            'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE',
            'ICUSTAY_ID', 'HCUP_CCS_2015', 'USE_IN_BENCHMARK'
        ],
        "index": None
    },
    "subject_icu_history": {
        "columns": [
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME',
            'LOS', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'DOB',
            'DOD', 'AGE', 'MORTALITY_INUNIT', 'MORTALITY', 'MORTALITY_INHOSPITAL'
        ],
        "index": None
    }
}

idx_to_file_type = {
    0: "timeseries",
    1: "episodic_data",
    2: "subject_events",
    3: "subject_diagnoses",
    4: "subject_icu_history"
}

FILE_TYPE_KEYS = ("timeseries", "episodic_data", "subject_events", "subject_diagnoses",
                  "subject_icu_history")

DTYPES = {
    file_type: DATASET_SETTINGS[file_index]["dtype"]
    for file_type, file_index in zip(FILE_TYPE_KEYS, [
        "timeseries",
        "episodic_data",
        "subject_events",
        "diagnosis",
        "icu_history",
    ])
}

READER_TO_FILE_TYPE = {
    "read_episodic_data": "episodic_data",
    "read_events": "subject_events",
    "read_diagnoses": "subject_diagnoses",
    "read_icu_history": "subject_icu_history"
}


def test_properties():
    # TODO! test dtypes
    tests_io("Test case properties for ExtractedSetReader", level=0)
    reader = ExtractedSetReader(Path(SEMITEMP_DIR, "extracted"))
    assert reader.root_path == Path(
        SEMITEMP_DIR, "extracted"
    ), f"Expected root path is {str(Path(SEMITEMP_DIR, 'extracted'))}, but reader root path is {str(reader.root_path)}"

    # Should not be able to set root path
    with pytest.raises(AttributeError) as error:
        reader.root_path = Path("test")
        assert error.info == "can't set attribute"
    tests_io("Root path is correctly set")

    assert reader.subject_ids == ground_truth_subject_ids, f"Subjects are not in the ground truth: {list(set(reader.subject_ids) - set(ground_truth_subject_ids))}\n Subjects are missing from the reader attribute: {list(set(ground_truth_subject_ids) - set(reader.subject_ids))}"

    # Should not be able to set subject ids
    with pytest.raises(AttributeError) as error:
        reader.subject_ids = []
        assert error.info == "can't set attribute"
    tests_io("Subject ids are correct")


def assert_dtypes(dataframe: pd.DataFrame, dtypes: dict):
    assert all([
        dataframe.dtypes[column] == "object"
        if dtype == "str" else dataframe.dtypes[column] == dtype  # Might be translated to obj
        for column, dtype in dtypes.items()
        if column in dataframe
    ])
    return True


def test_read_csv():
    # TODO! test dtypes
    tests_io("Test case read csv for ExtractedSetReader", level=0)
    reader = ExtractedSetReader(Path(SEMITEMP_DIR, "extracted"))

    # Test dtypes of different bites sizes
    dtype_mapping_template = {
        'ROW_ID': 'Int32',
        'SUBJECT_ID': 'Int64',
        'HADM_ID': 'Int32',
        'SEQ_NUM': 'Int8',
        'ICD9_CODE': 'str',
        'ICUSTAY_ID': 'float'
    }
    dtype_mapping = convert_dtype_dict(dtype_mapping_template, add_lower=False)

    absolute_diagnoses_path = Path(SEMITEMP_DIR, "extracted", "diagnoses.csv")
    absolute_df = reader.read_csv(absolute_diagnoses_path, dtypes=dtype_mapping)
    assert not absolute_df.empty, f"The file {str(absolute_diagnoses_path)} could not be found using absolute resolution!"
    assert_dtypes(absolute_df, dtype_mapping)

    relative_df = reader.read_csv("diagnoses.csv", dtypes=dtype_mapping)
    assert not relative_df.empty, f"The file {str(absolute_diagnoses_path)} could not be found using relative resolution!"
    assert_dtypes(relative_df, dtype_mapping)

    tests_io("Read CSV working with file name and absolute path")

    # Test for file types this is used on
    for file_name in ["episodic_info_df.csv", "icu_history.csv", "subject_info.csv"]:
        absolute_diagnoses_path = Path(SEMITEMP_DIR, "extracted", file_name)
        # Get dtypes
        settings_index_name = file_name.rstrip(".csv").rstrip("_df")
        dtypes = convert_dtype_dict(DATASET_SETTINGS[settings_index_name]["dtype"], add_lower=False)
        # Test absoulte read
        absolute_df = reader.read_csv(absolute_diagnoses_path, dtypes=dtypes)
        assert not absolute_df.empty, f"The file {str(absolute_diagnoses_path)} could not be found using absolute resolution!"
        assert_dtypes(absolute_df, dtypes)
        # Test relative read
        relative_df = reader.read_csv(file_name, dtypes=dtypes)
        assert not relative_df.empty, f"The file {str(absolute_diagnoses_path)} could not be found using relative resolution!"
        assert_dtypes(relative_df, dtypes)
        tests_io(f"Read CSV working correctly with {file_name}")
    tests_io("Read CSV tested successfully")


def test_read_timeseries():
    tests_io("Test case read timeseries for ExtractedSetReader", level=0)
    reader = ExtractedSetReader(Path(SEMITEMP_DIR, "extracted"))

    # --- test correct structure ---
    data = reader.read_timeseries(read_ids=True)
    # Make sure all subjects have stays dict
    assert all([isinstance(stays_dict, dict) for _, stays_dict in data.items()])

    # Make sure all indices are integers
    assert all([
        isinstance(subj_id, int) and isinstance(stay_id, int)
        for subj_id, stays_dict in data.items()
        for stay_id, _ in stays_dict.items()
    ])

    # Make sure all stays are frames
    assert all([
        isinstance(frame, pd.DataFrame)
        for _, stays_dict in data.items()
        for _, frame in stays_dict.items()
    ])
    # Ensure dtype correcteness
    assert all([
        assert_dtypes(frame, DTYPES["timeseries"])
        for _, stays_dict in data.items()
        for _, frame in stays_dict.items()
    ])
    # Make sure all frames are read
    assert all(
        [not frame.empty for _, stays_dict in data.items() for _, frame in stays_dict.items()])
    tests_io("Correct structure of timeseries data with ids")

    # --- test correct num subjects ---
    data = reader.read_timeseries(read_ids=True, num_subjects=10)

    # Make sure all indices are integers
    assert all([
        isinstance(subj_id, int) and isinstance(stay_id, int)
        for subj_id, stays_dict in data.items()
        for stay_id, _ in stays_dict.items()
    ])
    assert [isinstance(stays_dict, dict) for _, stays_dict in data.items()]
    # Make sure all stays are frames
    assert all([
        isinstance(frame, pd.DataFrame)
        for _, stays_dict in data.items()
        for _, frame in stays_dict.items()
    ])
    # Assert no empty frames
    assert all(
        [not frame.empty for _, stays_dict in data.items() for _, frame in stays_dict.items()])
    assert len(data) == 10
    assert all([
        assert_dtypes(frame, DTYPES["timeseries"])
        for _, stays_dict in data.items()
        for _, frame in stays_dict.items()
    ])
    tests_io("Correct num subjects when sepcified for timeseries data")

    data = reader.read_timeseries()
    assert len(data) == len(ground_truth_stay_ids)
    assert all([isinstance(frame, pd.DataFrame) for frame in data])
    assert all([not frame.empty for frame in data])
    assert all([assert_dtypes(frame, DTYPES["timeseries"]) for frame in data])
    tests_io("Correct dimension of timeseries data")
    tests_io("Timeseries read tested successfully")


@pytest.mark.parametrize(
    "reader_name", ["read_episodic_data", "read_events", "read_diagnoses", "read_icu_history"])
def test_read_remaining_file_types(reader_name: str):
    tests_io(f"Test case {reader_name} for ExtractedSetReader", level=0)
    reader = ExtractedSetReader(Path(SEMITEMP_DIR, "extracted"))
    reader_method = getattr(reader, reader_name)
    file_dtypes = DTYPES[READER_TO_FILE_TYPE[reader_name]]

    # --- test correct structure ---
    data = reader_method(read_ids=True)
    assert all([
        isinstance(subj_id, int)
        for subj_id, stays_dict in data.items()
        if isinstance(stays_dict, dict)
    ])
    # Make sure all subjects have stays dict
    assert [isinstance(stays_dict, pd.DataFrame) for _, stays_dict in data.items()]
    # Make sure all stays are frames
    assert all([not frame.empty for _, frame in data.items()])
    assert all([assert_dtypes(frame, file_dtypes) for frame in data.values()])
    tests_io(f"Correct structure of {reader_name} data with ids")

    # --- test correct num subjects ---
    data = reader_method(read_ids=True, num_subjects=10)
    assert [isinstance(stays_dict, pd.DataFrame) for _, stays_dict in data.items()]
    assert all([not frame.empty for _, frame in data.items()])
    assert all([assert_dtypes(frame, file_dtypes) for frame in data.values()])
    assert len(data) == 10
    tests_io(f"Correct num subjects when sepcified for {reader_name} data")

    data = reader_method()
    assert len(data) == len(ground_truth_subject_ids)
    assert all([not frame.empty for frame in data])
    assert all([assert_dtypes(frame, file_dtypes) for frame in data])
    tests_io(f"Correct dimension of {reader_name} data")
    tests_io(f"{reader_name} read tested successfully")


def test_read_subjects():
    tests_io("Test case read subjects for ExtractedSetReader", level=0)

    reader = ExtractedSetReader(Path(SEMITEMP_DIR, "extracted"))
    data_with_ids = reader.read_subjects(read_ids=True)

    # Make sure no subjects missing or additional
    gt_subject_ids = [
        int(directory.name)
        for directory in Path(TEST_GT_DIR, "extracted").iterdir()
        if directory.name.isnumeric()
    ]
    assert not set(gt_subject_ids) - set(data_with_ids.keys()) and not set(
        data_with_ids.keys()) - set(gt_subject_ids)
    tests_io("Correct dimension of dataset relative to subject ids")

    # Make sure no stays missing or additional
    gt_stay_ids = [
        re.findall('[0-9]+', file.name).pop()
        for directory in Path(TEST_GT_DIR, "extracted").iterdir() if directory.name.isnumeric()
        for file in directory.iterdir() if re.findall('[0-9]+', file.name)
    ]
    extracted_stay_ids = [
        str(stay_id)
        for subject_data in data_with_ids.values()
        for stay_id in subject_data["timeseries"].keys()
    ]
    assert not set(gt_stay_ids) - set(extracted_stay_ids) and \
           not set(extracted_stay_ids) - set(gt_stay_ids)
    tests_io("Correct dimension of dataset relative to stay ids")

    # Make sure the correct columns and indices are read for every file:
    for subject_data in data_with_ids.values():
        validate_subject_data(subject_data, file_type_keys=True, read_ids=True)

    data_without_ids = reader.read_subjects()
    assert len(data_without_ids) == len(gt_subject_ids)
    assert sum([len(subject_data["timeseries"]) for subject_data in data_without_ids
               ]) == len(extracted_stay_ids)

    for subject_data in data_without_ids:
        validate_subject_data(subject_data, file_type_keys=True, read_ids=False)

    data_without_keys = reader.read_subjects(file_type_keys=False)
    assert len(data_without_keys) == len(gt_subject_ids)
    assert sum([len(subject_data[0]) for subject_data in data_without_keys
               ]) == len(extracted_stay_ids)

    for subject_data in data_without_keys:
        validate_subject_data(subject_data, file_type_keys=False, read_ids=False)
    tests_io("Correct structure of dataset relative to file type keys")

    # Test the num samples option
    data_with_num_subjects = reader.read_subjects(read_ids=True, num_subjects=10)
    assert len(data_with_num_subjects) == 10
    for subject_data in data_with_num_subjects.values():
        validate_subject_data(subject_data, file_type_keys=True, read_ids=True)
    tests_io("Correct number of subjects when specified")

    # Test the subject ids option
    data_with_subject_ids = reader.read_subjects(read_ids=True,
                                                 subject_ids=["10006", "10011", "10036", "10088"])
    assert len(data_with_subject_ids) == 4
    assert list(data_with_subject_ids.keys()) == [10006, 10011, 10036, 10088]
    for subject_data in data_with_subject_ids.values():
        validate_subject_data(subject_data, file_type_keys=True, read_ids=True)
    tests_io("Correct subjects when specified")
    tests_io("Read subjects tested successfully")


def test_read_subjects_dir():
    tests_io("Test case read subjects ExtractedSetReader", level=0)
    reader = ExtractedSetReader(Path(SEMITEMP_DIR, "extracted"))
    ## With ids
    # Absolute
    data_with_ids = reader.read_subject(Path(reader.root_path, "10019"), read_ids=True)
    validate_subject_data(data_with_ids, file_type_keys=True, read_ids=True)
    # Relative
    data_with_ids = reader.read_subject("10019", read_ids=True)
    validate_subject_data(data_with_ids, file_type_keys=True, read_ids=True)
    tests_io("Correct relative and absolute resolution of subject data with ids")

    ## Without ids
    # Absolute
    data_without_ids = reader.read_subject(Path(reader.root_path, "10019"))
    validate_subject_data(data_without_ids, file_type_keys=True, read_ids=False)
    # Relative
    data_without_ids = reader.read_subject("10019")
    validate_subject_data(data_without_ids, file_type_keys=True, read_ids=False)
    tests_io("Correct relative and absolute resolution of subject data without ids")

    ## Without ids without file type keys
    # Absolute
    data_without_keys = reader.read_subject(Path(reader.root_path, "10019"), file_type_keys=False)
    validate_subject_data(data_without_keys, file_type_keys=False, read_ids=False)
    # Relative
    data_without_keys = reader.read_subject(Path(reader.root_path, "10019"), file_type_keys=False)
    validate_subject_data(data_without_keys, file_type_keys=False, read_ids=False)
    tests_io("Correct relative and absolute resolution of "
             "subject data without ids and file type keys")

    file_types = ("episodic_data", "subject_events", "subject_diagnoses", "subject_icu_history",
                  "timeseries")
    with pytest.raises(ValueError) as error:
        reader.read_subject(Path(reader.root_path, "10019"), file_types=("episodic_data"))
        assert error.value == f'file_types must be a tuple but is {type("")}'

    for _ in range(10):
        random_filetypes = random.sample(file_types, random.randint(1, 5))
        data = reader.read_subject("10019", file_types=random_filetypes, read_ids=True)
        validate_subject_data(data, file_type_keys=True, read_ids=True)

        data = reader.read_subject("10019", file_types=random_filetypes, read_ids=False)
        validate_subject_data(data, file_type_keys=True, read_ids=False)

        file_type_map = dict(enumerate(list(data.keys())))
        data = reader.read_subject("10019",
                                   file_types=random_filetypes,
                                   read_ids=False,
                                   file_type_keys=False)
        validate_subject_data(data,
                              file_type_keys=False,
                              read_ids=False,
                              file_type_mapping=file_type_map)
    tests_io("Correct file types subsampling when specified")
    tests_io("Read subjects tested successfully")


def validate_subject_data(subject_data,
                          file_type_keys,
                          read_ids,
                          file_type_mapping=idx_to_file_type):
    assert subject_data, "Subject data is empty!"
    # Make sure all file types are present in the data by checking their column names
    for idx, frame in subject_data.items() if file_type_keys else enumerate(subject_data):
        if not file_type_keys:
            idx = file_type_mapping[idx]
        if isinstance(frame, pd.DataFrame):
            filtered_colnames = [
                col for col in frame.columns if not col.isnumeric() and not col[1:].isnumeric()
            ]
            assert_dtypes(frame, DTYPES[idx])
            assert filtered_colnames == frame_properties[idx]["columns"]
            assert frame.index.name == frame_properties[idx]["index"]
        else:
            for value in frame.values() if read_ids else frame:
                filtered_colnames = [
                    col for col in value.columns if not col.isnumeric() and not col[1:].isnumeric()
                ]
                assert_dtypes(value, DTYPES[idx])
                assert filtered_colnames == frame_properties[idx]["columns"]
                assert value.index.name == frame_properties[idx]["index"]
    return


if __name__ == "__main__":
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    reader = datasets.load_data(chunksize=75835,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                task="PHENO")
    test_properties()
    test_read_csv()
    test_read_timeseries()
    for reader_mname in ["read_episodic_data", "read_events", "read_diagnoses", "read_icu_history"]:
        test_read_remaining_file_types(reader_mname)
    test_read_subjects_dir()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
