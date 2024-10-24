import json
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

    assert db._collection_name in db.db.list_collection_names()
    db.delete()
    assert db._collection_name not in db.db.list_collection_names()

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
        assert db._collection_name not in db.db._list_collection_names()

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
    assert db._collection_name.endswith('tests_data_semitemp_processed_DECOMP_progress')

    # Test with a path starting and ending with underscore
    db_path = Path("_invalid_start_and_end_")
    db = MongoDict(db_path)
    assert db._collection_name == "coll_invalid_start_and_end"
    db.delete()

    # Test with a very long name
    long_name = "a" * 100
    db = MongoDict(long_name)
    assert db._collection_name == "coll_" + "a" * 60
    assert len(db._collection_name) == 65
    db.delete()

    # Test with all invalid characters
    db_path = Path(TEMP_DIR, "/\\. \"$*<>:|?")
    db = MongoDict(db_path)
    assert db._collection_name == "coll_"
    db.delete()

    # Test actual database creation
    db_path = Path(TEMP_DIR, "test_db_creation")
    db = MongoDict(db_path, reinit=True)
    db['test_key'] = 'test_value'
    assert db._collection_name in db.db.list_collection_names()
    db.delete()

    tests_io("Tested db name sanitation successfully")

    # Test with empty string
    db = MongoDict("")
    assert db._collection_name == "coll_"
    db.delete()

    # Test with only invalid characters
    db = MongoDict("/\\. \"$*<>:|?")
    assert db._collection_name == "coll_"
    db.delete()

    # Test with a name that's exactly 63 characters after sanitization
    db = MongoDict("a" * 60 + "/\\. \"$*<>:|?")
    assert db._collection_name == "coll_" + "a" * 60
    assert len(db._collection_name) == 65
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


def test_collection_copy():
    tests_io("Starting test_collection_copy", level=0)

    # Create source database with multiple collections
    source_db_path = Path(TEMP_DIR, "source_db")

    # Ensure db does not exist
    MongoDict(source_db_path).delete()

    # Create source collections with different data
    source_collection1 = MongoDict(Path(source_db_path, "collection1"), reinit=True)
    source_collection1['test_key'] = 'test_value'
    source_collection1['nested'] = {'a': 1, 'b': 2}
    source_collection1.set_nested('deep.nested.key', 'value')

    source_collection2 = MongoDict(Path(source_db_path, "collection2"), reinit=True)
    source_collection2['different_key'] = 'different_value'

    # Create target database
    target_db_path = Path(TEMP_DIR, "target_db")

    # Ensure target db does not exist
    MongoDict(target_db_path).delete()

    target_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Test collection copy scenarios

    # Scenario 1: Copy to same collection name
    target_path = Path(target_db_path, "collection1")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'w') as f:
        f.write(source_collection1._collection_name)
    target_collection1 = MongoDict(target_path)
    assert target_collection1['test_key'] == 'test_value'
    assert target_collection1['nested'] == {'a': 1, 'b': 2}
    assert target_collection1['deep'] == {'nested': {'key': 'value'}}

    # Scenario 2: Copy to different collection name
    target_collection_new = MongoDict(Path(target_db_path, "new_collection"))
    source_collection1.copy(target_collection_new)

    assert target_collection_new['test_key'] == 'test_value'
    assert target_collection_new['nested'] == {'a': 1, 'b': 2}
    assert target_collection_new['deep'] == {'nested': {'key': 'value'}}

    # Scenario 3: Copy between different databases
    another_db_path = Path(TEMP_DIR, "another_db")
    another_collection = MongoDict(Path(another_db_path, "collection1"))
    source_collection1.copy(another_collection)

    assert another_collection['test_key'] == 'test_value'
    assert another_collection['nested'] == {'a': 1, 'b': 2}
    assert another_collection['deep'] == {'nested': {'key': 'value'}}

    # Scenario 4: Verify collections remain isolated
    target_collection2 = MongoDict(Path(target_db_path, "collection2"))
    source_collection2.copy(target_collection2)

    assert target_collection2['different_key'] == 'different_value'
    with pytest.raises(KeyError):
        _ = target_collection2['test_key']
    with pytest.raises(KeyError):
        _ = target_collection1['different_key']

    # Scenario 5: Test copying empty collection
    empty_source = MongoDict(Path(source_db_path, "empty_collection"))
    empty_target = MongoDict(Path(target_db_path, "empty_target"))
    empty_source.copy(empty_target)

    assert len(list(empty_target.collection.find())) == 0

    # Scenario 6: Test overwriting existing collection
    existing_target = MongoDict(Path(target_db_path, "existing"))
    existing_target['old_key'] = 'old_value'

    source_collection1.copy(existing_target)
    assert existing_target['test_key'] == 'test_value'  # Should have source data
    with pytest.raises(KeyError):
        _ = existing_target['old_key']  # Old data should be gone

    # Clean up
    source_collection1.delete()
    source_collection2.delete()
    target_collection1.delete()
    target_collection2.delete()
    target_collection_new.delete()
    another_collection.delete()
    empty_source.delete()
    empty_target.delete()
    existing_target.delete()

    tests_io("Collection copy tests passed successfully")


def test_json_fallback():
    tests_io("Starting test_json_fallback", level=0)

    # Create test data
    test_data = {
        "subjects": {
            "total": 7,
            "10112": {
                "total": 1,
                "224063": 1
            },
            "10119": {
                "total": 1,
                "247686": 1
            },
            "10069": {
                "total": 1,
                "290490": 1
            },
            "40601": {
                "total": 1,
                "279529": 1
            }
        }
    }

    db_path = Path(TEMP_DIR, "json_test_db")

    # Ensure db does not exist
    db = MongoDict(db_path)
    db.delete()

    # Create JSON file
    db_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{db_path}.json")
    with open(json_path, 'w') as f:
        json.dump(test_data, f)

    # Initialize database from JSON with specific collection
    db = MongoDict(db_path)

    # Verify data was loaded correctly
    assert db['subjects']['total'] == 7
    assert db['subjects']['10112']['total'] == 1
    assert db['subjects']['10112']['224063'] == 1
    assert db['subjects']['10119']['247686'] == 1
    assert db['subjects']['10069']['290490'] == 1
    assert db['subjects']['40601']['279529'] == 1

    # Verify the complete structure
    assert db['subjects'] == {
        "total": 7,
        "10112": {
            "total": 1,
            "224063": 1
        },
        "10119": {
            "total": 1,
            "247686": 1
        },
        "10069": {
            "total": 1,
            "290490": 1
        },
        "40601": {
            "total": 1,
            "279529": 1
        }
    }

    # Clean up
    db.delete()
    json_path.unlink()

    tests_io("JSON fallback tests passed successfully")


def test_fallback_priority():
    tests_io("Starting test_fallback_priority", level=0)

    # Create source with specific collection
    source_collection = "source_collection"
    source_db = MongoDict(collection_name=source_collection, reinit=True)
    source_db['source_key'] = 'source_value'

    # Create progress file
    target_db_path = Path(TEMP_DIR, "priority_target_db")
    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_db_path, 'w') as f:
        f.write(source_db._db_name)

    # Create JSON file with different data and collection
    json_data = {'json_key': 'json_value'}
    json_path = Path(f"{target_db_path}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f)

    # Initialize with same collection name - should prefer copying from source
    target_db = MongoDict(collection_name=source_collection)

    # Verify correct collection was used
    assert target_db['source_key'] == 'source_value'
    with pytest.raises(KeyError):
        _ = target_db['json_key']

    # Test with different collection name - should create new collection
    different_collection = "different_collection"
    different_db = MongoDict(collection_name=different_collection)

    # Clean up
    source_db.delete()
    target_db.delete()
    different_db.delete()

    tests_io("Fallback priority tests passed successfully")


def test_error_handling():
    tests_io("Starting test_error_handling", level=0)

    # Test with invalid progress file and specific collection
    collection_name = "error_collection"

    MongoDict(collection_name=collection_name).delete()
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    invalid_path = Path(TEMP_DIR, "invalid_db")
    with open(invalid_path, 'w') as f:
        f.write("nonexistent_db")

    # Should create new empty collection when source doesn't exist
    db = MongoDict(collection_name=collection_name)
    db['test'] = 'value'  # Should work with new empty collection

    # Test with invalid JSON file
    json_path = Path(TEMP_DIR, "invalid_db.json")
    with open(json_path, 'w') as f:
        f.write("invalid json content")

    # Should handle invalid JSON gracefully and create new collection
    new_collection = "new_collection"
    db2 = MongoDict(collection_name=new_collection)
    db2['test2'] = 'value2'  # Should work with new empty collection

    # Clean up
    db.delete()
    db2.delete()

    tests_io("Error handling tests passed successfully")


def test_multiple_collections():
    tests_io("Starting test_multiple_collections", level=0)

    # Create multiple collections
    collection1 = MongoDict(collection_name="collection1")
    collection2 = MongoDict(collection_name="collection2")

    # Test data isolation between collections
    collection1['key'] = 'value1'
    collection2['key'] = 'value2'

    assert collection1['key'] == 'value1'
    assert collection2['key'] == 'value2'
    assert collection1.collection.name != collection2.collection.name

    # Test copying between collections
    collection3 = MongoDict(collection_name="collection3")
    collection1.copy(collection3)

    assert collection3['key'] == 'value1'

    # Clean up
    collection1.delete()
    collection2.delete()
    collection3.delete()

    tests_io("Multiple collections tests passed successfully")


if __name__ == "__main__":
    import shutil
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_init_and_delete()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_json_fallback()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_collection_copy()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
    test_fallback_priority()
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
    test_error_handling()
    if TEMP_DIR.exists():
        shutil.rmtree(str(TEMP_DIR))
