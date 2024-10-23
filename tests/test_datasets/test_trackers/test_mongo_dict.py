import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from utils.IO import *
from storable import MongoDict
from tests.tsettings import *


def test_set_nested():
    tests_io("Starting test_set_nested", level=0)
    with MongoDict(Path(TEMP_DIR, "progress"), reinit=True) as db:
        db.set_nested('a', 1)
        assert db["a"] == 1
        tests_io("Simple set test passed")

        db.set_nested('b.c.d', 2)
        assert db['b'] == {'c': {'d': 2}}
        tests_io("Nested set new test passed")

        db.set_nested('e.f', 3)
        db.set_nested('e.g', 4)
        assert db['e'] == {'f': 3, 'g': 4}
        tests_io("Nested set existing test passed")

        db.set_nested('e.h.n', 4)
        assert db['e'] == {'f': 3, 'g': 4, 'h': {'n': 4}}
        tests_io("Nested set existing test passed")

        db.set_nested('h.i', 5)
        db.set_nested('h.i.j', 6)
        assert db['h'] == {'i': {'j': 6}}
        tests_io("Nested set overwrite test passed")

        db.set_nested('k.l.m.n.o', 7)
        assert db['k'] == {'l': {'m': {'n': {'o': 7}}}}
        tests_io("Deeply nested set test passed")

        db.set_nested('p.q.r', 8)
        db.set_nested('p.q.s', 9)
        assert db['p'] == {'q': {'r': 8, 's': 9}}
        tests_io("Update existing nested test passed")

        with pytest.raises(ValueError):
            db.set_nested('', 10)
        tests_io("Set with empty key test passed")

        db.set_nested('t.u', None)
        assert db['t']['u'] is None
        tests_io("Set with None value test passed")

        complex_value = {'x': [1, 2, 3], 'y': {'z': 'test'}}
        db.set_nested('v.w', complex_value)
        assert db['v']['w'] == complex_value
        tests_io("Set with complex value test passed")

    tests_io("All tests in test_set_nested passed successfully")


def test_init_and_delete():
    tests_io("Starting test_init_and_delete", level=0)
    db_path = Path(TEMP_DIR, "test_init")

    # Test initialization and deletion
    db = MongoDict(db_path)

    # Insert some data to ensure the database is created
    db['test_key'] = 'test_value'

    assert db._db_name in db.client.list_database_names()
    db.delete()
    assert db._db_name not in db.client.list_database_names()

    # Test reinit parameter
    db = MongoDict(db_path, reinit=True)
    db['test'] = 'value'
    db = MongoDict(db_path, reinit=True)
    with pytest.raises(KeyError):
        _ = db['test']

    db.delete()
    tests_io("All tests in test_init_and_delete passed successfully")


def test_contains_and_getitem():
    tests_io("Starting test_contains_and_getitem", level=0)
    with MongoDict(Path(TEMP_DIR, "test_contains"), reinit=True) as db:
        db['key'] = 'value'
        assert 'key' in db
        assert db['key'] == 'value'
        assert 'nonexistent' not in db
        with pytest.raises(KeyError):
            _ = db['nonexistent']
    tests_io("All tests in test_contains_and_getitem passed successfully")


def test_update():
    tests_io("Starting test_update", level=0)
    with MongoDict(Path(TEMP_DIR, "test_update"), reinit=True) as db:
        db['a'] = {'b': 1, 'c': 2}
        db.update('a', {'c': 3, 'd': 4})
        assert db['a'] == {'b': 1, 'c': 3, 'd': 4}

        db.update('new_key', {'x': 10})
        assert db['new_key'] == {'x': 10}
    tests_io("All tests in test_update passed successfully")


def test_clear():
    tests_io("Starting test_clear", level=0)
    with MongoDict(Path(TEMP_DIR, "test_clear"), reinit=True) as db:
        db['a'] = 1
        db['b'] = 2
        db.clear()
        assert 'a' not in db
        assert 'b' not in db
    tests_io("All tests in test_clear passed successfully")


def test_get_nested():
    tests_io("Starting test_get_nested", level=0)
    with MongoDict(Path(TEMP_DIR, "test_get_nested"), reinit=True) as db:
        db.set_nested('a.b.c', 1)
        assert db.get_nested('a.b.c') == 1
        assert db.get_nested('a.b') == {'c': 1}
        with pytest.raises(KeyError):
            db.get_nested('a.b.d')
        with pytest.raises(KeyError):
            db.get_nested('nonexistent')
    tests_io("All tests in test_get_nested passed successfully")


def test_items_and_to_dict():
    tests_io("Starting test_items_and_to_dict", level=0)
    with MongoDict(Path(TEMP_DIR, "test_items"), reinit=True) as db:
        test_data = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': [4, 5, 6]}
        for k, v in test_data.items():
            db[k] = v

        # Test items()
        items = dict(db.items())
        assert items == test_data

        # Test to_dict()
        assert db.to_dict() == test_data
    tests_io("All tests in test_items_and_to_dict passed successfully")


def test_method_integration():
    tests_io("Starting comprehensive NestedMongoDict test", level=0)
    db_path = Path(TEMP_DIR, "comprehensive_test")

    with MongoDict(db_path, reinit=True) as db:
        # Test initialization and basic operations

        # Test set operations
        db['simple_key'] = 'simple_value'
        assert db._db_name in db.client.list_database_names()
        db.set_nested('nested.key.path', 'nested_value')
        db.set_nested('another.nested.path', {'complex': 'value'})

        # Test get operations
        assert db['simple_key'] == 'simple_value'
        assert db.get_nested('nested.key.path') == 'nested_value'
        assert db.get_nested('another.nested.path') == {'complex': 'value'}

        # Test contains
        assert 'simple_key' in db
        assert 'nested' in db
        assert 'nonexistent' not in db

        db.update('nested', {'key': {'path': 'updated_nested_value'}})
        assert db.get_nested('nested.key.path') == 'updated_nested_value'

        # Test complex nested operations
        db.set_nested('complex.path.with.many.levels', [1, 2, 3])
        assert db.get_nested('complex.path.with.many.levels') == [1, 2, 3]

        # TODO! cool idea but for the future
        # db.set_nested('complex.path.with.many.levels.0', 'overwritten')
        # assert db.get_nested('complex.path.with.many.levels') == ['overwritten', 2, 3]

        # Test overwriting behavior
        db.set_nested('overwrite.test', 'initial')
        db.set_nested('overwrite.test.nested', 'should_overwrite')
        assert db.get_nested('overwrite.test') == {'nested': 'should_overwrite'}

        # Test items and to_dict
        items_dict = dict(db.items())
        to_dict_result = db.to_dict()
        assert items_dict == to_dict_result
        assert 'simple_key' in items_dict
        assert 'nested' in items_dict
        assert 'complex' in items_dict

        # Test clear
        db.clear()
        assert db.to_dict() == {}

        # Repopulate for further testing
        db['key1'] = 'value1'
        db['key2'] = {'nested': 'value2'}

        # Test print and print_all (we'll just check if they run without errors)
        db.print()
        db.print_all()

        # Test delete
        db.delete()
        assert db._db_name not in db.client.list_database_names()

    # Test if the context manager properly closed the connection
    with pytest.raises(Exception):
        db.client.server_info()

    tests_io("Comprehensive NestedMongoDict test completed successfully", level=0)


def test_numpy_pandas_types():
    tests_io("Starting test_numpy_pandas_types", level=0)
    db_path = Path(TEMP_DIR, "test_numpy_pandas")

    with MongoDict(db_path, reinit=True) as db:
        # Test numpy types
        db['np_int'] = np.int64(42)
        db['np_float'] = np.float32(3.14)
        db['np_array'] = np.array([1, 2, 3, 4, 5])

        # Test nested structure with mixed types
        db['mixed'] = {
            'np_value': np.float64(2.718),
            'normal_list': [1, 2, 3],
            'nested_dict': {
                'np_array': np.array([10, 20, 30]),
            }
        }

        # Verify numpy types
        assert isinstance(db['np_int'], int)
        assert db['np_int'] == 42
        assert isinstance(db['np_float'], float)
        assert abs(db['np_float'] - 3.14) < 1e-6
        assert isinstance(db['np_array'], list)
        assert db['np_array'] == [1, 2, 3, 4, 5]

        # Verify nested structure
        mixed = db['mixed']
        assert isinstance(mixed['np_value'], float)
        assert abs(mixed['np_value'] - 2.718) < 1e-6
        assert mixed['normal_list'] == [1, 2, 3]
        assert isinstance(mixed['nested_dict']['np_array'], list)
        assert mixed['nested_dict']['np_array'] == [10, 20, 30]

        # TODO! potentially extend to frames, series etc.

    tests_io("All tests in test_numpy_pandas_types passed successfully")


def test_db_name_handling():
    tests_io("Starting test_db_sanitization_and_collection_handling", level=0)

    # Test with a long path
    db_path = Path(TEMP_DIR, "workdir/tests/data/semitemp/processed/DECOMP_progress")
    db = MongoDict(db_path)
    assert db._db_name == 'db_paces_pymimic3_tests_data_semitemp_processed_DECOMP_progress'

    # Test with a path starting and ending with underscore
    db_path = Path("_invalid_start_and_end_")
    db = MongoDict(db_path, collection_name="system.collection")
    assert db._db_name == "db_invalid_start_and_end"
    db.delete()

    # Test with a very long name
    long_name = "a" * 100
    db = MongoDict(long_name)
    assert db._db_name == "db_" + "a" * 60
    assert len(db._db_name) == 63
    db.delete()

    # Test with all invalid characters
    db_path = Path(TEMP_DIR, "/\\. \"$*<>:|?")
    db = MongoDict(db_path)
    assert db._db_name == "db_"
    db.delete()

    # Test actual database creation
    db_path = Path(TEMP_DIR, "test_db_creation")
    db = MongoDict(db_path, collection_name="test_collection", reinit=True)
    db['test_key'] = 'test_value'
    assert db._db_name in db.client.list_database_names()
    db.delete()

    tests_io("Tested db name sanitation successfully")

    # Test with empty string
    db = MongoDict("")
    assert db._db_name == "db_"
    db.delete()

    # Test with only invalid characters
    db = MongoDict("/\\. \"$*<>:|?")
    assert db._db_name == "db_"
    db.delete()

    # Test with a name that's exactly 63 characters after sanitization
    db = MongoDict("a" * 60 + "/\\. \"$*<>:|?")
    assert db._db_name == "db_" + "a" * 60
    assert len(db._db_name) == 63
    db.delete()

    tests_io("All tests in test_edge_cases passed successfully")


def test_nested_delete():
    tests_io("Starting test_nested_delete", level=0)

    db_path = Path(TEMP_DIR, "test_nested_delete")

    with MongoDict(db_path, reinit=True) as db:
        # Set up a nested structure
        db['a'] = {'b': {'c': {'d': 1, 'e': 2}, 'f': 3}, 'g': 4}

        # Test deleting a deeply nested key
        assert db.delete_nested('a.b.c.d') == True
        assert db['a'] == {'b': {'c': {'e': 2}, 'f': 3}, 'g': 4}

        # Test deleting a key that makes parent empty
        assert db.delete_nested('a.b.c.e') == True
        assert db['a'] == {'b': {'f': 3}, 'g': 4}

        # Test deleting a non-existent key
        assert db.delete_nested('a.b.c.x') == False
        assert db['a'] == {'b': {'f': 3}, 'g': 4}

        # Test deleting a key that empties multiple levels
        db['x'] = {'y': {'z': {}}}
        assert db.delete_nested('x.y.z') == True
        assert 'x' not in db

        # Test deleting a top-level key
        assert db.delete_nested('a.g') == True
        assert db['a'] == {'b': {'f': 3}}

        # Test deleting the last nested key
        assert db.delete_nested('a.b.f') == True
        assert 'a' not in db

    tests_io("All tests in test_nested_delete passed successfully")


if __name__ == "__main__":
    import shutil
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_nested_delete()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_db_name_handling()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_set_nested()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_init_and_delete()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_contains_and_getitem()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_update()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_clear()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_get_nested()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_items_and_to_dict()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_method_integration()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_numpy_pandas_types()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
