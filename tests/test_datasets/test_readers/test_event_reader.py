import shutil
import numpy as np
import pandas as pd
import datasets
import pytest
from time import sleep
from pathlib import Path
from datasets.trackers import ExtractionTracker
from datasets.readers import EventReader
from tests.tsettings import *
from tests.pytest_utils.general import assert_dataframe_equals
from utils.IO import *
from settings import *

DTYPES = DATASET_SETTINGS["subject_events"]["dtype"]

columns = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM"]


def assert_dtypes(dataframe: pd.DataFrame):
    assert all([
        dataframe.dtypes[column] == "object"
        if dtype == "str" else dataframe.dtypes[column] == dtype  # Might be translated to obj
        for column, dtype in DTYPES.items()
        if column in dataframe
    ])
    return True


def test_get_full_chunk():
    tests_io("Test case getting full chunk", level=0)
    tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    event_reader = EventReader(chunksize=900000, dataset_folder=TEST_DATA_DEMO, tracker=tracker)

    samples, frame_lengths = event_reader.get_chunk()
    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == len(frame)
        assert len(frame) == frame_lengths[csv_name]
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)
    tests_io("Test case getting full chunk succeeded")


def test_get_full_chunk():
    tests_io("Test case getting full chunk", level=0)
    tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    event_reader = EventReader(chunksize=900000, dataset_folder=TEST_DATA_DEMO, tracker=tracker)

    samples, frame_lengths = event_reader.get_chunk()
    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == len(frame)
        assert len(frame) == frame_lengths[csv_name]
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)
    tests_io("Test case getting full chunk succeeded")


def test_get_mutlitple_chunks():
    tests_io("Test case getting multiple chunks", level=0)

    tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    event_reader = EventReader(chunksize=1000, dataset_folder=TEST_DATA_DEMO, tracker=tracker)

    samples, frame_lengths = event_reader.get_chunk()

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 1000
        assert frame_lengths[csv_name] == 1000
        assert frame.index[-1] == 999
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    samples, frame_lengths = event_reader.get_chunk()

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 1000
        assert frame_lengths[csv_name] == 1000
        assert frame.index[0] == 1000
        assert frame.index[-1] == 1999
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    tests_io("Test case getting multiple chunks succeeded")


def test_resume_get_chunk():
    tests_io("Test case resuming get_chunk", level=0)
    orig_tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    orig_event_reader = EventReader(chunksize=1000,
                                    dataset_folder=TEST_DATA_DEMO,
                                    tracker=orig_tracker)

    samples, frame_lengths = orig_event_reader.get_chunk()

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 1000
        assert frame_lengths[csv_name] == 1000
        assert frame.index[0] == 0
        assert frame.index[-1] == 999
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    # Tracker register the frames as read
    orig_tracker.start_event_rows += frame_lengths

    restored_tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"))
    restored_event_reader = EventReader(chunksize=200,
                                        dataset_folder=TEST_DATA_DEMO,
                                        tracker=restored_tracker)

    samples, frame_lengths = restored_event_reader.get_chunk()

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 200
        assert frame_lengths[csv_name] == 200
        assert frame.index[0] == 1000
        assert frame.index[-1] == 1199
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    # Event producer only read 200 samples then crashed
    orig_tracker.start_event_rows += {
        "CHARTEVENTS.csv": 200,
        "LABEVENTS.csv": 200,
        "OUTPUTEVENTS.csv": 200
    }

    restored_tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"))
    restored_event_reader = EventReader(chunksize=500,
                                        dataset_folder=TEST_DATA_DEMO,
                                        tracker=restored_tracker)

    # Resume reading after crash from same chunk
    samples, frame_lengths = restored_event_reader.get_chunk()

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 500
        assert frame_lengths[csv_name] == 500
        assert frame.index[0] == 1200
        assert frame.index[-1] == 1699
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    tests_io("Test case resuming get_chunk succeeded")


def test_switch_chunk_sizes():
    tests_io("Test case switching chunk sizes", level=0)
    orig_tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    orig_event_reader = EventReader(chunksize=1000,
                                    dataset_folder=TEST_DATA_DEMO,
                                    tracker=orig_tracker)

    samples, frame_lengths = orig_event_reader.get_chunk()
    orig_tracker.start_event_rows += frame_lengths

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 1000
        assert frame_lengths[csv_name] == 1000
        assert frame.index[0] == 0
        assert frame.index[-1] == 999
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    # Crash and resume with half the chunk size
    orig_tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    orig_event_reader = EventReader(chunksize=500,
                                    dataset_folder=TEST_DATA_DEMO,
                                    tracker=orig_tracker)

    samples, frame_lengths = orig_event_reader.get_chunk()
    orig_tracker.start_event_rows += frame_lengths

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 500
        assert frame_lengths[csv_name] == 500
        assert frame.index[0] == 1000
        assert frame.index[-1] == 1499
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    # Crash and resume with half the chunk size
    orig_tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    orig_event_reader = EventReader(chunksize=700,
                                    dataset_folder=TEST_DATA_DEMO,
                                    tracker=orig_tracker)

    samples, frame_lengths = orig_event_reader.get_chunk()
    orig_tracker.start_event_rows += frame_lengths

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 700
        assert frame_lengths[csv_name] == 700
        assert frame.index[0] == 1500
        assert frame.index[-1] == 2199
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    # Crash and resume with half the chunk size
    orig_tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    orig_event_reader = EventReader(chunksize=100,
                                    dataset_folder=TEST_DATA_DEMO,
                                    tracker=orig_tracker)

    samples, frame_lengths = orig_event_reader.get_chunk()
    orig_tracker.start_event_rows += frame_lengths

    assert len(samples) == 3
    for csv_name, frame in samples.items():
        assert len(frame) == 100
        assert frame_lengths[csv_name] == 100
        assert frame.index[0] == 2200
        assert frame.index[-1] == 2299
        assert not (set(frame.columns) - set(columns))
        assert not (set(columns) - set(frame.columns))
        assert_dtypes(frame)

    tests_io("Test case switching chunk sizes succeeded")


@pytest.mark.parametrize("subject_ids", [
    [42458],
    [42458, 41976],
    [10112, 10119, 40601, 42458, 10111],
])
def test_subject_ids(subject_ids: list):
    """Test if the event reader stops on last subject occurence.
    """
    tests_io("Test case with subject ids", level=0)
    tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    subject_ids = [42458, 41976]  # int version 41976
    event_reader = EventReader(chunksize=1000,
                               subject_ids=subject_ids,
                               dataset_folder=TEST_DATA_DEMO,
                               tracker=tracker)
    # Takes only a second on my machine
    sleep(2)
    frame_lengths = True
    previous_samples = {}
    test_csv = ["CHARTEVENTS.csv", "LABEVENTS.csv", "OUTPUTEVENTS.csv"]

    sample_buffer = {"CHARTEVENTS.csv": list(), "LABEVENTS.csv": list(), "OUTPUTEVENTS.csv": list()}

    counter = 0
    while not event_reader.done_reading:
        samples, frame_lengths = event_reader.get_chunk()
        for csv in frame_lengths:
            if frame_lengths[csv]:
                previous_samples[csv] = samples[csv]
                sample_buffer[csv].append(samples[csv])
        counter += 1
    assert sample_buffer["CHARTEVENTS.csv"]
    for csv in previous_samples.keys():
        last_occurence = event_reader._last_occurrence[csv]
        last_sample = previous_samples[csv].index[-1]

        # Test if all events have been read
        assert np.isclose(
            last_occurence, last_sample, atol=1
        ) and last_occurence >= last_sample, f"Last occurence line: {last_occurence} and last sample line: {last_sample}"
        tests_io(
            f"{csv}: Last occurence line: {last_occurence} and last sample line: {last_sample}")
        test_csv.remove(csv)

    all_data = event_reader.get_all()

    all_data = all_data[all_data["SUBJECT_ID"].isin(subject_ids)]
    all_data = all_data.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID"])
    all_data = all_data.reset_index(drop=True)
    chunk_data = pd.concat([pd.concat(buffer) for buffer in sample_buffer.values()])
    chunk_data = chunk_data.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID"])
    chunk_data = chunk_data.reset_index(drop=True)
    assert_dataframe_equals(chunk_data.astype("object"), all_data.astype("object"))
    assert_dtypes(chunk_data)
    assert_dtypes(all_data)

    assert not test_csv, f"Test csvs not empty: {test_csv}"
    tests_io("Test case with subject ids succeeded")


def test_compare_full_to_iterative():
    """Test if the event reader stops on last subject occurence.
    """
    tests_io("Test case with subject ids", level=0)
    tracker = ExtractionTracker(Path(TEMP_DIR, "extracted", "progress"), num_samples=None)
    event_reader = EventReader(chunksize=1000, dataset_folder=TEST_DATA_DEMO, tracker=tracker)
    # Takes only a second on my machine
    sleep(2)
    frame_lengths = True
    previous_samples = {}
    test_csv = ["CHARTEVENTS.csv", "LABEVENTS.csv", "OUTPUTEVENTS.csv"]

    sample_buffer = {"CHARTEVENTS.csv": list(), "LABEVENTS.csv": list(), "OUTPUTEVENTS.csv": list()}

    counter = 0
    while not event_reader.done_reading:
        samples, frame_lengths = event_reader.get_chunk()
        for csv in frame_lengths:
            if frame_lengths[csv]:
                previous_samples[csv] = samples[csv]
                sample_buffer[csv].append(samples[csv])
        counter += 1
    assert sample_buffer["CHARTEVENTS.csv"]

    all_data = event_reader.get_all()

    all_data = all_data.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID"])
    all_data = all_data.reset_index(drop=True)
    chunk_data = pd.concat([pd.concat(buffer) for buffer in sample_buffer.values()])
    chunk_data = chunk_data.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID"])
    chunk_data = chunk_data.reset_index(drop=True)
    assert_dataframe_equals(chunk_data.astype("object"), all_data.astype("object"))
    assert_dtypes(chunk_data)
    assert_dtypes(all_data)

    tests_io("Test case with subject ids succeeded")


if __name__ == '__main__':
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_compare_full_to_iterative()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_resume_get_chunk()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_compare_full_to_iterative()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    for subject_ids in [[42458], [42458, 41976], [10112, 10119, 40601, 42458, 10111]]:
        test_subject_ids(subject_ids)
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_get_full_chunk()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_get_mutlitple_chunks()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_switch_chunk_sizes()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
