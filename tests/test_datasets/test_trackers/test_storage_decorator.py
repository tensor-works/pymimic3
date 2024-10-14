from tests.tsettings import *
import pytest
import shutil
from pathlib import Path
from utils.IO import *
# from storable import storable
from storable import storable


@storable
class TestClass:
    num_samples: int = 0
    time_elapsed: float = 0.0
    finished: bool = False
    subjects: dict = {"a": 0, "b": 0}
    names: dict = {"a": "a", "b": "b"}


@storable
class CountTestClass:
    num_samples: int = 0
    time_elapsed: float = 0.0
    finished: bool = False
    subject_ids: list = list()
    subjects: dict = {"a": 0, "b": 0}
    _store_total: bool = True


@storable
class IntKeyTestClass:
    int_key_dict: dict = {1: "one", 2: "two"}
    mixed_key_dict: dict = {1: "one", "two": 2, 3: "three"}


# Test the storable
def test_storable_basics():
    tests_io("Test case basic capabilities of Storable.", level=0)

    # Test the default values
    assert TestClass.num_samples == 0
    assert TestClass.time_elapsed == 0.0
    assert TestClass.finished == False
    assert TestClass.subjects == {"a": 0, "b": 0}
    assert TestClass.names == {"a": "a", "b": "b"}

    # Test the correct recreation of its attributes with the correct types
    check_dtypes(TestClass)
    tests_io("Succeeded testing initialization.")

    # Test the storage path
    test_instance = TestClass(Path(TEMP_DIR, "progress"))

    assert Path(TEMP_DIR, "progress").is_file()

    # Test assignment
    test_instance.num_samples = 10
    test_instance.time_elapsed = 1.0
    test_instance.finished = True
    test_instance.subjects = {"a": 1, "b": 2}
    test_instance.names = {"a": "b", "b": "a"}

    assert test_instance.num_samples == 10
    assert test_instance.time_elapsed == 1.0
    assert test_instance.finished == True
    assert test_instance.subjects == {"a": 1, "b": 2}
    assert test_instance.names == {"a": "b", "b": "a"}

    check_dtypes(test_instance)
    tests_io("Succeeded testing assignment.")

    # Test restorable assignment
    del test_instance
    test_instance = TestClass(Path(TEMP_DIR, "progress"))

    assert test_instance.num_samples == 10
    assert test_instance.time_elapsed == 1.0
    assert test_instance.finished == True
    assert test_instance.subjects == {"a": 1, "b": 2}
    assert test_instance.names == {"a": "b", "b": "a"}

    check_dtypes(test_instance)

    tests_io("Succeeded testing restoration.")

    # Test __iadd__
    test_instance.num_samples += 10
    test_instance.time_elapsed += 1.1

    assert test_instance.num_samples == 20
    assert test_instance.time_elapsed == 2.1

    check_dtypes(test_instance)

    # Test restorable __iadd__
    del test_instance
    test_instance = TestClass(Path(TEMP_DIR, "progress"))

    assert test_instance.num_samples == 20
    assert test_instance.time_elapsed == 2.1
    check_dtypes(test_instance)

    tests_io("Succeeded testing __iadd__.")


def test_dictionary_iadd():
    tests_io("Test case dictionary iadd.", level=0)
    test_instance = TestClass(Path(TEMP_DIR, "progress"))
    test_instance.subjects == {"a": 0, "b": 0}

    test_instance.subjects += {"a": 1, "b": 2}

    assert test_instance.subjects == {"a": 1, "b": 2}

    test_instance.subjects += {"a": 1}
    assert test_instance.subjects == {"a": 2, "b": 2}

    tests_io("Succeeded testing numerical dictionary iadd.")

    # Test restorable dictionary iadd
    del test_instance
    test_instance = TestClass(Path(TEMP_DIR, "progress"))

    assert test_instance.subjects == {"a": 2, "b": 2}
    tests_io("Succeeded testing restoration of numerical dictionary iadd.")


def test_dictionary_update():
    tests_io("Test case dictionary update.", level=0)
    test_instance = TestClass(Path(TEMP_DIR, "progress"))
    test_instance.names = {"a": {"a": 1, "b": 1}, "b": {"a": 2, "b": 2}}
    test_instance.names.update({"a": {"c": 1, "d": 1}, "c": {"a": 1, "b": 1}})
    assert test_instance.names == {
        "a": {
            "a": 1,
            "b": 1,
            "c": 1,
            "d": 1
        },
        "b": {
            "a": 2,
            "b": 2
        },
        "c": {
            "a": 1,
            "b": 1
        }
    }
    tests_io("Succeeded testing dictionary update.")
    del test_instance
    test_instance = TestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.names == {
        "a": {
            "a": 1,
            "b": 1,
            "c": 1,
            "d": 1
        },
        "b": {
            "a": 2,
            "b": 2
        },
        "c": {
            "a": 1,
            "b": 1
        }
    }
    tests_io("Succeeded testing restoration of dictionary update.")


def test_total_count():
    # Only implemented for single nestation and not for iadd
    tests_io("Test case total count for storable decorator.", level=0)
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))

    test_instance.subjects = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
    assert test_instance.subjects == {
        "a": {
            "a": 1,
            "b": 2,
            "total": 3
        },
        "b": {
            "a": 2,
            "b": 4,
            "total": 6
        },
        "total": 9
    }
    tests_io("Succeeded dictionary assignment total count.")
    test_instance.subjects.update({"a": {"c": 3}})
    assert test_instance.subjects == {
        "a": {
            "a": 1,
            "b": 2,
            "c": 3,
            "total": 6
        },
        "b": {
            "a": 2,
            "b": 4,
            "total": 6
        },
        "total": 12
    }
    tests_io("Succeeded dictionary nested update total count.")

    test_instance.subjects = {"a": 0, "b": 0}
    assert test_instance.subjects == {"a": 0, "b": 0, "total": 0}
    tests_io("Succeeded dictionary reassignment total count.")

    test_instance.subjects = {}
    assert test_instance.subjects == {"total": 0}
    tests_io("Succeeded dictionary empty reassignment total count.")

    test_instance.subjects.update({"a": 1, "b": 2})
    assert test_instance.subjects == {"a": 1, "b": 2, "total": 3}
    tests_io("Succeeded dictionary simple data type overwrite total count.")

    test_instance.subjects.update({"a": 1, "b": 1})
    assert test_instance.subjects == {"a": 1, "b": 1, "total": 2}

    test_instance.subjects.update({"a": 2, "b": 2})
    assert test_instance.subjects == {"a": 2, "b": 2, "total": 4}

    test_instance.subjects.update({"c": 1, "d": 1})
    assert test_instance.subjects == {"a": 2, "b": 2, "c": 1, "d": 1, "total": 6}

    test_instance.subjects.update({"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}})
    assert test_instance.subjects == {
        "a": {
            "a": 1,
            "b": 2,
            "total": 3
        },
        "b": {
            "a": 2,
            "b": 4,
            "total": 6
        },
        "c": 1,
        "d": 1,
        "total": 11
    }
    tests_io("Succeeded dictionary complex data type overwrite total count.")
    test_instance.subjects.update({"c": 3})
    assert test_instance.subjects == {
        "a": {
            "a": 1,
            "b": 2,
            "total": 3
        },
        "b": {
            "a": 2,
            "b": 4,
            "total": 6
        },
        "c": 3,
        "d": 1,
        "total": 13
    }

    test_instance.subjects.update({"d": {"a": 1, "b": 1}})
    assert test_instance.subjects == {
        "a": {
            "a": 1,
            "b": 2,
            "total": 3
        },
        "b": {
            "a": 2,
            "b": 4,
            "total": 6
        },
        "c": 3,
        "d": {
            "a": 1,
            "b": 1,
            "total": 2
        },
        "total": 14
    }


def test_list_basic():
    tests_io("Test case basic list for storable decorator.", level=0)
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))

    test_instance.subject_ids = ["a", "b", "c"]
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]

    tests_io("Succeeded testing basic list assignment.")
    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]
    tests_io("Succeeded testing restoration of basic list assignment.")
    # We are tempering with the cls so need to make sure nothing is permanent
    del test_instance
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == []
    # assert test_instance._progress["subject_ids"] == []
    tests_io("Succeeded testing list non permanency on database deletion.")


def test_list_extend():
    tests_io("Test case extend list for storable decorator.", level=0)
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))

    test_instance.subject_ids.extend(["a", "b", "c"])
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]
    tests_io("Succeeded testing extend list assignment.")

    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]
    tests_io("Succeeded testing restoration of extend list assignment.")

    test_instance.subject_ids.extend(["d", "e", "f"])
    assert test_instance.subject_ids == ["a", "b", "c", "d", "e", "f"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c", "d", "e", "f"]
    tests_io("Succeeded testing extend list assignment.")

    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b", "c", "d", "e", "f"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c", "d", "e", "f"]
    tests_io("Succeeded testing restoration of extend list assignment.")


def test_list_pop():
    tests_io("Test case pop list for storable decorator.", level=0)
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))

    # Assign and make sure
    test_instance.subject_ids = ["a", "b", "c"]
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]

    # Restore to see if permament
    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]

    # Now pop
    popper = test_instance.subject_ids.pop()
    assert popper == "c"
    assert test_instance.subject_ids == ["a", "b"]
    # assert test_instance._progress["subject_ids"] == ["a", "b"]
    tests_io("Succeeded testing pop list assignment.")
    # Restore to see if permament
    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b"]
    # assert test_instance._progress["subject_ids"] == ["a", "b"]
    tests_io("Succeeded testing restoration of pop list assignment.")


def test_list_remove():
    tests_io("Test case remove list for storable decorator.", level=0)
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))

    # Assign and make sure
    test_instance.subject_ids = ["a", "b", "c"]
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]

    # Restore to see if permament
    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]

    # Now remove
    test_instance.subject_ids.remove("b")
    assert test_instance.subject_ids == ["a", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "c"]
    tests_io("Succeeded testing remove list assignment.")

    # Restore to see if permament
    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "c"]
    tests_io("Succeeded testing restoration of remove list assignment.")


def test_list_append():
    tests_io("Test case append list for storable decorator.", level=0)
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))

    # Assign and make sure
    test_instance.subject_ids = ["a", "b", "c"]
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]

    # Restore to see if permament
    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b", "c"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c"]

    # Now append
    test_instance.subject_ids.append("d")
    assert test_instance.subject_ids == ["a", "b", "c", "d"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c", "d"]
    tests_io("Succeeded testing append list assignment.")

    # Restore to see if permament
    del test_instance
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))
    assert test_instance.subject_ids == ["a", "b", "c", "d"]
    # assert test_instance._progress["subject_ids"] == ["a", "b", "c", "d"]
    tests_io("Succeeded testing restoration of append list assignment.")


def test_int_keys():
    tests_io("Test case for integer keys in storable decorator.", level=0)

    # Create an instance and modify the dictionaries
    test_instance = IntKeyTestClass(Path(TEMP_DIR, "int_key_test"))
    test_instance.int_key_dict[3] = "three"
    test_instance.mixed_key_dict[4] = "four"

    # Check immediate state
    assert test_instance.int_key_dict == {1: "one", 2: "two", 3: "three"}
    assert test_instance.mixed_key_dict == {1: "one", "two": 2, 3: "three", 4: "four"}
    assert all(isinstance(key, int) for key in test_instance.int_key_dict.keys())
    assert all(isinstance(key, (int, str)) for key in test_instance.mixed_key_dict.keys())
    tests_io("Succeeded testing immediate state after modification.")

    # Delete the instance and create a new one to test persistence
    del test_instance
    restored_instance = IntKeyTestClass(Path(TEMP_DIR, "int_key_test"))

    # Check restored state
    assert restored_instance.int_key_dict == {1: "one", 2: "two", 3: "three"}
    assert restored_instance.mixed_key_dict == {1: "one", "two": 2, 3: "three", 4: "four"}
    assert all(isinstance(key, int) for key in restored_instance.int_key_dict.keys())
    assert all(isinstance(key, (int, str)) for key in restored_instance.mixed_key_dict.keys())
    tests_io("Succeeded testing restored state after reloading.")

    # Test nested dictionaries with integer keys
    restored_instance.int_key_dict[4] = {1: "nested_one", 2: "nested_two"}
    del restored_instance
    nested_instance = IntKeyTestClass(Path(TEMP_DIR, "int_key_test"))

    assert nested_instance.int_key_dict[4] == {1: "nested_one", 2: "nested_two"}
    assert all(isinstance(key, int) for key in nested_instance.int_key_dict[4].keys())
    tests_io("Succeeded testing nested dictionaries with integer keys.")

    # Test updating with integer keys
    nested_instance.int_key_dict.update({5: "five", 6: "six"})
    assert nested_instance.int_key_dict[5] == "five" and nested_instance.int_key_dict[6] == "six"
    assert all(isinstance(key, int) for key in nested_instance.int_key_dict.keys())
    tests_io("Succeeded testing dictionary update with integer keys.")

    # Test deleting with integer keys
    del nested_instance.int_key_dict[1]
    assert 1 not in nested_instance.int_key_dict
    tests_io("Succeeded testing deletion of integer keys.")

    tests_io("All integer key tests passed successfully.")


def update_instance(instance, iterations, thread_id):
    print(f"Thread {thread_id} starting. Initial state: {instance.__dict__}")
    for i in range(iterations):
        instance.num_samples += 1
        instance.time_elapsed += 0.1
        instance.subject_ids.append(random.randint(1, 1000))
        key = random.choice(['a', 'b'])
        instance.subjects[key] += 1
    instance.finished = True


import pytest
from concurrent.futures import ThreadPoolExecutor
import time
import random


def test_concurrent_access():
    test_instance = CountTestClass(Path(TEMP_DIR, "progress"))

    num_threads = 5
    iterations_per_thread = 100

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(update_instance, test_instance, iterations_per_thread, i)
            for i in range(num_threads)
        ]

        # Wait for all threads to complete
        for future in futures:
            future.result()

    print(f"All threads completed. Final state: {test_instance.__dict__}")

    # Verify the results
    assert test_instance.num_samples == num_threads * iterations_per_thread, f"Expected {num_threads * iterations_per_thread}, got {test_instance.num_samples}"
    assert pytest.approx(test_instance.time_elapsed,
                         abs=0.1) == num_threads * iterations_per_thread * 0.1
    assert len(
        test_instance.subject_ids
    ) == num_threads * iterations_per_thread, f"Expected {num_threads * iterations_per_thread}, got {len(test_instance.subject_ids)}"
    assert test_instance.subjects[
        "total"] == num_threads * iterations_per_thread, f"Expected {num_threads * iterations_per_thread}, got {sum(test_instance.subjects.values())}"
    assert test_instance.finished


def test_persistence(test_instance):
    print(f"Persistence test starting. Original instance state: {test_instance.__dict__}")
    # Verify persistence by creating a new instance
    new_instance = CountTestClass()
    print(f"New instance created for persistence test. State: {new_instance.__dict__}")

    assert new_instance.num_samples == test_instance.num_samples, f"num_samples mismatch: {new_instance.num_samples} != {test_instance.num_samples}"
    assert pytest.approx(
        new_instance.time_elapsed, abs=0.1
    ) == test_instance.time_elapsed, f"time_elapsed mismatch: {new_instance.time_elapsed} != {test_instance.time_elapsed}"
    assert new_instance.subject_ids == test_instance.subject_ids, f"subject_ids mismatch: {new_instance.subject_ids} != {test_instance.subject_ids}"
    assert new_instance.subjects == test_instance.subjects, f"subjects mismatch: {new_instance.subjects} != {test_instance.subjects}"
    assert new_instance.finished == test_instance.finished, f"finished mismatch: {new_instance.finished} != {test_instance.finished}"


def check_dtypes(instance):
    assert isinstance(instance.num_samples, int)
    assert isinstance(instance.time_elapsed, float)
    assert isinstance(instance.finished, bool)
    assert isinstance(instance.subjects, dict)
    assert isinstance(instance.names, dict)


if __name__ == "__main__":
    # if TEMP_DIR.is_dir():
    #     shutil.rmtree(str(TEMP_DIR))
    # test_concurrent_access()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_list_basic()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_total_count()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_list_extend()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_list_append()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_list_pop()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_list_remove()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    test_total_count()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    test_storable_basics()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    test_dictionary_iadd()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    test_dictionary_update()
    if TEMP_DIR.is_dir():
        shutil.rmtree(str(TEMP_DIR))

    tests_io("All tests passed")
