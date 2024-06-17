# TODO! subject ids may be string or numbers this might have some effect on the tracker
import shutil
from pathlib import Path
from datasets.trackers import PreprocessingTracker
from utils.IO import *
from tests.tsettings import *

tracker_state = {"subjects": {}, "finished": False, "num_subjects": None}


def test_processing_tracker_basics():
    tests_io("Test case basic capabilities of PreprocessingTracker.", level=0)
    # Create an instance of PreprocessingTracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))

    # Test correct initialization
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress
        if attribute == "subjects":
            assert getattr(tracker, attribute) == {"total": 0}
        else:
            assert getattr(tracker, attribute) == value

    # Assignment attriubutes
    tracker.subjects = {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4
        }
    }
    assert tracker._progress["subjects"] == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "total": 3
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "total": 9
    }
    # Test correct assignment
    assert tracker.subjects == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "total": 3
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "total": 9
    }
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)

    tests_io("Succeeded testing initialization.")

    # Test correct restoration after assignment
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress

    assert tracker.subjects == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "total": 3
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "total": 9
    }
    assert tracker.num_subjects == None
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)
    tests_io("Succeeded testing restoration.")

    # Test custom update implementation
    tracker.subjects.update({"subject_3": {"stay_1": 3, "stay_2": 6}})
    assert tracker._progress["subjects"] == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "total": 3
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "subject_3": {
            "stay_1": 3,
            "stay_2": 6,
            "total": 9
        },
        "total": 18
    }
    tracker.subjects == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "total": 3
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "subject_3": {
            "stay_1": 3,
            "stay_2": 6,
            "total": 9
        },
        "total": 18
    }
    tracker.subjects.update({"subject_1": {"stay_3": 3}})
    assert tracker._progress["subjects"] == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "stay_3": 3,
            "total": 6
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "subject_3": {
            "stay_1": 3,
            "stay_2": 6,
            "total": 9
        },
        "total": 21
    }
    tracker.subjects == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "stay_3": 3,
            "total": 6
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "subject_3": {
            "stay_1": 3,
            "stay_2": 6,
            "total": 9
        },
        "total": 21
    }
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2", "subject_3"])
    assert not set(["subject_1", "subject_2", "subject_3"]) - set(tracker.subject_ids)

    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress
    assert tracker.subjects == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "stay_3": 3,
            "total": 6
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "subject_3": {
            "stay_1": 3,
            "stay_2": 6,
            "total": 9
        },
        "total": 21
    }
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2", "subject_3"])
    assert not set(["subject_1", "subject_2", "subject_3"]) - set(tracker.subject_ids)
    tests_io("Succeeded testing custom __iadd__ implementation.")


def test_finishing_mechanism():
    tests_io("Test case finishing mechanism of PreprocessingTracker.", level=0)
    # Create an instance of PreprocessingTracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))

    # Test correct initialization
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress
        if attribute == "subjects":
            assert getattr(tracker, attribute) == {"total": 0}
        else:
            assert getattr(tracker, attribute) == value

    # Simulate processing of two subjects
    tracker.subjects.update({
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4
        }
    })
    # Make sure subjects are set
    assert tracker.subjects == {
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
            "total": 3
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
            "total": 6
        },
        "total": 9
    }
    # Finish tracker
    tracker.is_finished = True
    assert tracker.is_finished == True
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)
    for key, stays in tracker.subjects.items():
        if key == "total":
            continue
        # Make sure total length is set per subject
        assert stays["total"] == sum([count for key, count in stays.items() if key != "total"])
    tests_io("Succeeded testing total length creation.")

    # Test correct restoration after finishing
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    assert tracker.is_finished == True
    for key, stays in tracker.subjects.items():
        if key == "total":
            continue
        # Make sure total length is set per subject
        assert stays["total"] == sum([count for key, count in stays.items() if key != "total"])
    tests_io("Succeeded testing restoration after finishing.")


def test_num_subject_option():
    tests_io("Test case num_subjects option of PreprocessingTracker.", level=0)
    # Create an instance of PreprocessingTracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_num_subjects(2)

    # Test correct initialization
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress
        if attribute == "num_subjects":
            assert getattr(tracker, attribute) == 2
        elif attribute == "subjects":
            assert getattr(tracker, attribute) == {"total": 0}
        else:
            assert getattr(tracker, attribute) == value

    # Simulate processing of two subjects
    tracker.subjects.update({
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2,
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4,
        }
    })
    tracker.is_finished = True

    # Test decrease of num_subjects
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_num_subjects(1)
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress

    # Make sure state is restored and finished is set to True
    assert tracker.num_subjects == 2
    assert tracker.is_finished == True
    assert len(tracker.subjects) - 1 == 2
    assert tracker.subjects["total"] == 9
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)

    tests_io("Succeeded testing decrease of num_subjects.")

    # Test increase to original num_subjects
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_num_subjects(2)
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress

    # Make sure state is restored and finished is set to True
    assert tracker.num_subjects == 2
    assert tracker.is_finished == True
    assert len(tracker.subjects) - 1 == 2
    assert tracker.subjects["total"] == 9
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)

    tests_io("Succeeded testing increase to original num_subjects.")

    # Test increase above original num_subjects
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_num_subjects(3)
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress

    # Make sure state is restored and finished is set to True
    assert tracker.num_subjects == 3
    assert tracker.is_finished == False
    assert len(tracker.subjects) - 1 == 2
    assert tracker.subjects["total"] == 9
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)

    tests_io("Succeeded testing increase above original num_subjects.")

    # Simulate processing of one more subject
    tracker.subjects.update({"subject_3": {"stay_1": 3, "stay_2": 6}})
    tracker.is_finished = True
    assert tracker.num_subjects == 3
    assert tracker.is_finished == True
    assert len(tracker.subjects) - 1 == 3
    assert tracker.subjects["subject_3"]["total"] == 9
    assert tracker.subjects["total"] == 18
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2", "subject_3"])
    assert not set(["subject_1", "subject_2", "subject_3"]) - set(tracker.subject_ids)
    # Test switch to None
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_num_subjects(None)
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress

    # Make sure state is restored and finished is set to False
    assert tracker.num_subjects == None
    assert tracker.is_finished == False
    assert len(tracker.subjects) - 1 == 3
    assert tracker.subjects["total"] == 18
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2", "subject_3"])
    assert not set(["subject_1", "subject_2", "subject_3"]) - set(tracker.subject_ids)
    tests_io("Succeeded testing switch to None.")


def test_subject_ids_option():
    tests_io("Test case subject_ids option of PreprocessingTracker.", level=0)
    # Create an instance of PreprocessingTracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    # Test correct initialization
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress
        if attribute == "subjects":
            assert getattr(tracker, attribute) == {"total": 0}
        else:
            assert getattr(tracker, attribute) == value

    # Simulate processing of two subjects
    tracker.subjects.update({
        "subject_1": {
            "stay_1": 1,
            "stay_2": 2
        },
        "subject_2": {
            "stay_1": 2,
            "stay_2": 4
        }
    })
    tracker.is_finished = True

    # Test truthy subject_ids
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_subject_ids(["subject_1", "subject_2"])

    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress

    # All necessary subjects are processed
    assert tracker.is_finished == True
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)

    # Test decrease of num_subjects
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_subject_ids(["subject_1"])

    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress
    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)
    assert tracker.is_finished == True
    tests_io("Succeeded testing truthy subject_ids.")

    # Test falsey subject_ids
    del tracker
    tracker = PreprocessingTracker(storage_path=Path(TEMP_DIR, "progress"))
    tracker.set_subject_ids(["subject_1", "subject_2", "subject_3"])
    for attribute, value in tracker_state.items():
        assert attribute in tracker._progress

    assert not set(tracker.subject_ids) - set(["subject_1", "subject_2"])
    assert not set(["subject_1", "subject_2"]) - set(tracker.subject_ids)
    assert tracker.is_finished == False
    tests_io("Succeeded testing falsey subject_ids.")


if __name__ == "__main__":
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    test_processing_tracker_basics()
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    test_finishing_mechanism()
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    test_num_subject_option()
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    test_subject_ids_option()
    tests_io("Succeeded testing PreprocessingTracker.")
