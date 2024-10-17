import pytest
import shutil
from datasets.trackers import ExtractionTracker
from pathlib import Path
from utils.IO import *
from tests.tsettings import *

TRACKER_STATE = {
    "start_event_rows": {
        "OUTPUTEVENTS.csv": 0,
        "LABEVENTS.csv": 0,
        "CHARTEVENTS.csv": 0
    },
    "has_episodic_data": False,
    "has_timeseries": False,
    "subject_ids": list(),
    "has_subject_events": False,
    "count_total_samples": 0,
    "has_icu_history": False,
    "has_diagnoses": False,
    "has_bysubject_info": False,
    "finished": False,
    "num_samples": None
}

EVENT_BOOLS = [
    "has_episodic_data", "has_subject_events", "has_timeseries", "has_bysubject_info", "is_finished"
]


def test_extraction_tracker_basics():
    tests_io("Test case basic capabilities of ExtractionTracker.", level=0)
    # Create an instance of ExtractionTracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_samples=None)

    # Test correct initialization
    for attribute, value in TRACKER_STATE.items():
        assert getattr(tracker, attribute) == value

    tests_io("Succeeded testing initialization.")
    # Assignment bools and nums
    for attribute, value in TRACKER_STATE.items():
        # These states influence the state of the tracker from within the ExtractionTracker __init__
        # and are therefore not part of basic funcitonalities
        if attribute not in ["start_event_rows", "subject_ids", "num_samples", "num_subjects"]:
            setattr(tracker, attribute,
                    True if isinstance(getattr(tracker, attribute), bool) else 10)

    tracker.start_event_rows = {"OUTPUTEVENTS.csv": 10, "LABEVENTS.csv": 10, "CHARTEVENTS.csv": 10}

    # Test correct assignment
    for attribute, value in TRACKER_STATE.items():
        if attribute not in ["start_event_rows", "subject_ids", "num_samples", "num_subjects"]:
            assert getattr(
                tracker,
                attribute) == (True if isinstance(getattr(tracker, attribute), bool) else 10)

    assert tracker.start_event_rows == {
        "OUTPUTEVENTS.csv": 10,
        "LABEVENTS.csv": 10,
        "CHARTEVENTS.csv": 10
    }
    tests_io("Succeeded testing assignment.")

    # Test correct restoration after assignment
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_samples=None)
    for attribute, value in TRACKER_STATE.items():
        if attribute not in ["start_event_rows", "subject_ids", "num_samples", "num_subjects"]:
            assert getattr(
                tracker,
                attribute) == (True if isinstance(getattr(tracker, attribute), bool) else 10)

    assert tracker.start_event_rows == {
        "OUTPUTEVENTS.csv": 10,
        "LABEVENTS.csv": 10,
        "CHARTEVENTS.csv": 10
    }
    tests_io("Succeeded testing restoration.")

    # Test correct __iadd__ implementation
    for attribute, value in TRACKER_STATE.items():
        if isinstance(getattr(tracker, attribute), (int, float)):
            setattr(tracker, attribute,
                    getattr(tracker, attribute) +
                    10)  #This is how iadd is perfomed for simple properties

    tracker.start_event_rows += {"OUTPUTEVENTS.csv": 10, "LABEVENTS.csv": 10, "CHARTEVENTS.csv": 10}

    for attribute, value in TRACKER_STATE.items():
        if isinstance(attribute, (int, float)):
            assert getattr(tracker, attribute) == 20

    assert tracker.start_event_rows == {
        "OUTPUTEVENTS.csv": 20,
        "LABEVENTS.csv": 20,
        "CHARTEVENTS.csv": 20
    }

    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_samples=None)

    for attribute, value in TRACKER_STATE.items():
        if isinstance(attribute, (int, float)):
            assert getattr(tracker, attribute) == 20

    assert tracker.start_event_rows == {
        "OUTPUTEVENTS.csv": 20,
        "LABEVENTS.csv": 20,
        "CHARTEVENTS.csv": 20
    }
    tests_io("Succeeded testing __iadd__ implementation.")


def test_num_samples_option():
    # Test the logic of increasing and decreasing num sapmles upon reinstantiation
    tests_io("Test case sample target option of ExtractionTracker.", level=0)
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_samples=10)

    assert tracker.num_samples == 10
    for attribute, value in TRACKER_STATE.items():
        if attribute != "num_samples":
            assert getattr(tracker, attribute) == value

    # Simulate extraction done
    for attribute in EVENT_BOOLS:
        setattr(tracker, attribute, True)
    tracker.count_total_samples = 10

    # Test decreasing the number of samples upon reinstantiation
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_samples=5)
    # Asser total samples unchanged
    assert tracker.count_total_samples == 10

    # Assert still done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == True

    tests_io("Succeeded testing sample target reduction")
    # Test increasing the number of samples upon reinstantiation
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_samples=15)
    # Asser total samples unchanged
    assert tracker.count_total_samples == 10

    # Assert not done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == False

    # Check set correctly
    assert tracker.num_samples == 15

    tests_io("Succeeded testing sample target increase")

    # Simulate extraction done
    for attribute in EVENT_BOOLS:
        setattr(tracker, attribute, True)
    tracker.count_total_samples = 15
    # Test setting sample_target to None
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_samples=None)
    # Check init
    for attribute, value in TRACKER_STATE.items():
        if attribute not in EVENT_BOOLS + ["num_samples", "count_total_samples"]:
            assert getattr(tracker, attribute) == value

    # Check extraction done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == False

    # Check event read not done
    assert tracker.num_samples is None
    tests_io("Succeeded testing sample target set to None")


def test_num_subjects_option():
    # Test the logic of increasing and decreasing num sapmles upon reinstantiation
    tests_io("Test case sample target option of ExtractionTracker.", level=0)
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_subjects=10)
    # Check init
    assert tracker.num_subjects == 10
    for attribute, value in TRACKER_STATE.items():
        if attribute != "num_subjects":
            assert getattr(tracker, attribute) == value

    # Simulate extraction done
    for attribute in EVENT_BOOLS:
        setattr(tracker, attribute, True)
    tracker.subject_ids.extend(list(range(10)))

    # Test decreasing the number of samples upon reinstantiation
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_subjects=5)
    # Assert subject ids unchanged
    assert tracker.subject_ids == list(range(10))
    # Assert done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == True

    tests_io("Succeeded testing num subject reduction")

    # Test increasing the number of samples upon reinstantiation
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_subjects=15)
    # Assert subject ids unchanged
    assert tracker.subject_ids == list(range(10))
    # Assert done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == False
    tracker.subject_ids.extend(list(range(10, 15)))

    tests_io("Succeeded testing num subject increase")

    # Simulate extraction done
    for attribute in EVENT_BOOLS:
        setattr(tracker, attribute, True)

    # Test setting sample_target to None
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), num_subjects=None)
    # Check init
    for attribute, value in TRACKER_STATE.items():
        if attribute not in EVENT_BOOLS + ["num_samples", "subject_ids"]:
            assert getattr(tracker, attribute) == value

    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == False
    # Assert subject ids unchanged
    assert tracker.subject_ids == list(range(15))

    tests_io("Succeeded testing num subject set to None")


def test_subject_ids_option():
    # Test the logic of passing subject ids to the tracker
    # If all ids have already been extracted is_finished should be set to True
    # If some ids have not yet been extracted is_finished should be set to False
    # The tracker does not know all subject ids in advance, so it cannot provide
    # the list of still to be processed subjects
    tests_io("Test case subject ids option of ExtractionTracker.", level=0)

    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"),
                                subject_ids=list(range(10)))
    # Check init
    for attribute, value in TRACKER_STATE.items():
        if attribute != "subject_ids":
            assert getattr(tracker, attribute) == value

    # Simulate extraction done
    for attribute in EVENT_BOOLS:
        setattr(tracker, attribute, True)
    tracker.count_total_samples = 10
    tracker.subject_ids.extend(list(range(10)))

    # Test decreasing the number of subjects upon reinstantiation
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), subject_ids=list(range(5)))
    # Assert subject ids unchanged
    assert tracker.count_total_samples == 10
    assert tracker.subject_ids == list(range(10))
    # Assert done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == True

    tests_io("Succeeded testing subject ids reduction")
    # Test increasing the number of subjects upon reinstantiation
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"),
                                subject_ids=list(range(15)))
    # Check init
    for attribute, value in TRACKER_STATE.items():
        if attribute not in EVENT_BOOLS + ["num_samples", "count_total_samples", "subject_ids"]:
            assert getattr(tracker, attribute) == value
    # Assert subject ids unchanged
    assert tracker.count_total_samples == 10
    assert tracker.subject_ids == list(range(10))
    # Assert not done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == False

    tracker.subject_ids.extend(list(range(10, 15)))
    tracker.count_total_samples += 5
    assert tracker.subject_ids == list(range(15))
    assert tracker.start_event_rows == {
        "OUTPUTEVENTS.csv": 0,
        "LABEVENTS.csv": 0,
        "CHARTEVENTS.csv": 0
    }

    # Test setting subject_ids to None
    del tracker
    tracker = ExtractionTracker(storage_path=Path(TEMP_DIR, "progress"), subject_ids=None)
    # Check init
    for attribute, value in TRACKER_STATE.items():
        if attribute not in EVENT_BOOLS + ["num_samples", "count_total_samples", "subject_ids"]:
            assert getattr(tracker, attribute) == value
    # Assert subject ids unchanged
    assert tracker.count_total_samples == 15
    assert tracker.subject_ids == list(range(15))

    # Assert not done
    for attribute in EVENT_BOOLS:
        assert getattr(tracker, attribute) == False

    assert tracker.start_event_rows == {
        "OUTPUTEVENTS.csv": 0,
        "LABEVENTS.csv": 0,
        "CHARTEVENTS.csv": 0
    }


if __name__ == "__main__":
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_subject_ids_option()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_num_subjects_option()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_extraction_tracker_basics()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_num_samples_option()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    print("All tests passed!")
