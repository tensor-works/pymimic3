import os
from pymongo import MongoClient
from typing import Any, Dict, Tuple, List
from functools import wraps
from copy import deepcopy
from pathlib import Path
from storable.proxy_types import ProxyValue, ProxyProperty
from storable.mongo_dict import MongoDict


def storable(cls):
    """
    A class decorator that adds persistence functionality to a class.

    This decorator modifies the class to automatically save its attributes
    to a SQLite database and load them when instantiated.

    Parameters
    ----------
    cls : type
        The class to be decorated.

    Returns
    -------
    type
        The decorated class with added persistence functionality.

    Notes
    -----
    The decorated class will have additional methods for managing the 
    persistent storage of its state.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> 
    >>> @storable
    ... @dataclass
    ... class UserStats:
    ...     username: str
    ...     posts: dict
    ...     likes: dict
    ...     store_total: bool = True # (set to False to disable total computations)
    ... 
    >>> # Create a new user
    >>> user = UserStats("alice", storage_path="alice_stats.db")
    >>> 
    >>> # Add some data
    >>> user.posts["day1"] = 5
    >>> user.likes["day1"] = 10
    >>> 
    >>> # Data is automatically persisted
    >>> del user
    >>> 
    >>> # Load user data in a new session
    >>> loaded_user = UserStats("alice", storage_path="alice_stats.db")
    >>> print(f"Posts: {loaded_user.posts}, Likes: {loaded_user.likes}")
    Posts: {'day1': 5, 'total': 5}, Likes: {'day1': 10, 'total': 10}
    >>> 
    >>> # Update data
    >>> loaded_user.posts["day2"] = 3
    >>> loaded_user.likes["day2"] = 7
    >>> 
    >>> # Check updated totals
    >>> print(f"Total posts: {loaded_user.posts['total']}, Total likes: {loaded_user.likes['total']}")
    Total posts: 8, Total likes: 17
    >>> 
    >>> # Data persists across sessions
    >>> del loaded_user
    >>> final_check = UserStats("alice", storage_path="alice_stats.db")
    >>> print(f"Final state - Posts: {final_check.posts}, Likes: {final_check.likes}")
    Final state - Posts: {'day1': 5, 'day2': 3, 'total': 8}, Likes: {'day1': 10, 'day2': 7, 'total': 17}
    """
    # When you think about it this is highly illegal but the only way of keeping them cls attributes unchanged between instantiations.
    # Find a better way if you can, you are the real hero.
    originals = {
        name: deepcopy(attr)
        for name, attr in vars(cls).items()
        if not name.startswith("_") and isinstance(attr, (int, float, bool, dict, list, type(None)))
    }
    setattr(cls, '_originals', originals)
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        """
        Initialize the storable class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._progress = {}
        for name, original_value in cls._originals.items():
            setattr(cls, name, deepcopy(original_value))

        if "storage_path" in kwargs:
            self._path = Path(kwargs.pop('storage_path'))
        elif args:
            self._path = Path(args[0])
            args = tuple(args[1:])
        else:
            raise ValueError(
                "No storage path provided to storable. Either provide as first positional argument or with the storage_path keyword argument."
            )

        self._path.parent.mkdir(exist_ok=True, parents=True)

        self._save_frequency = kwargs.pop('save_frequency', 1)
        self._access_count = 0
        self._lock_file = Path(self._path.parent, f"{self._path.stem}.lock")
        # lockutils.set_defaults(lock_path=str(self._lock_file))
        self._file_lock = None

        # Load or initialize progress
        if self.db_exists(self._path):
            if not self._path.exists():
                self.delete_db(self._path)
                self._wrap_attributes()
                self._write(self._progress)
                with open(self._path, 'w') as file:
                    file.write(str(self._path))
            else:
                self._progress = self._read()
                self._wrap_attributes()
        else:
            self._wrap_attributes()
            self._write(self._progress)
            with open(self._path, 'w') as file:
                file.write(str(self._path))

        original_init(self, *args, **kwargs)

    def _get_callback(self, key, db=None):
        if db is None:
            with MongoDict(self._path) as db:
                ret = db.get_nested(key)
                return ret
        else:
            return db.get_nested(key)

    def _set_callback(self, key, value, db=None):
        if db is None:
            with MongoDict(self._path) as db:
                if isinstance(value, ProxyValue):
                    value = value._value
                db.set_nested(key, value)
                #db[key] = value
        else:
            if isinstance(value, ProxyValue):
                value = value._value
            #db[key] = value
            db.set_nested(key, value)

    def _del_callback(self, key, db=None):
        if db is None:
            with MongoDict(self._path) as db:
                db.delete_nested(key)
        else:
            db.delete_nested(key)

    def _update_dict(self, attribute, updates, db=None):
        if db is None:
            with MongoDict(self._path) as db:
                db.update(attribute, updates)
        else:
            db.update(attribute, updates)

    def _write(self, update_dict, db=None):
        if db is None:
            with MongoDict(self._path) as db:
                for k, value in update_dict.items():
                    if isinstance(value, ProxyValue):
                        value = value._value
                    db[k] = value
        else:
            for k, value in update_dict.items():
                if isinstance(value, ProxyValue):
                    value = value._value
                db[k] = value

    def _read(self, db=None):
        if db is None:
            with MongoDict(self._path) as db:
                ret = db.to_dict()
                return ret
        else:
            return db.to_dict()

    def _db_keys(self, db=None):
        if db is None:
            with MongoDict(self._path) as db:
                return list(db.keys())
        else:
            return list(db.keys())

    def _progress_keys(self):
        return list(set([key.split("^")[0] for key in self._db_keys()]))

    def print_db(self):
        print("===== Printing =====")
        with MongoDict(self._path) as db:
            for key, value in db.items():
                print(f'{key}: {value}')
        print("===== End Printing =====")

    def db_exists(self, coll_name, host=os.getenv("MONGODB_HOST"), port=None):
        coll_name = MongoDict._sanitize_collection_name(coll_name)
        client = MongoClient(host, port)
        try:
            # Get the list of all database names
            coll_list = client[os.getenv("MONGODB_NAME", "workspace")].list_collection_names()
            return coll_name in coll_list
        finally:
            client.close()

    def delete_db(self, coll_name, host=os.getenv("MONGODB_HOST"), port=None):
        # Delete the specified database
        coll_name = MongoDict._sanitize_collection_name(coll_name)
        client = MongoClient(host, port)
        try:
            db = client[os.getenv("MONGODB_NAME", "workspace")]
            if coll_name in db.list_collection_names():
                db.drop_collection(coll_name)
        finally:
            client.close()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._wrap_attributes()

    cls.__setstate__ = __setstate__
    cls.__init__ = __init__
    cls._set_callback = _set_callback
    cls._get_callback = _get_callback
    cls._del_callback = _del_callback
    cls._write = _write
    cls._read = _read
    cls._db_keys = _db_keys
    cls._progress_keys = _progress_keys
    cls._update_dict = _update_dict
    cls.print_db = print_db
    cls.db_exists = db_exists
    cls.delete_db = delete_db

    # Attribute wrapping logic
    def _wrap_attributes(self):
        """
        Wrap the attributes of the class with the appropriate descriptors.
        """

        for name, attr in vars(cls).items():
            if (isinstance(attr, (int, float, bool, dict, list)) or
                    attr is None) and not name.startswith("_"):
                attr = self._progress.get(name, attr)
                if isinstance(attr, dict):
                    # Store total is a mess but necessary for preprocessing trackers
                    store_total = getattr(cls, "_store_total", False)
                    # TODO! store total
                    setattr(cls, name, ProxyProperty(name, attr, store_total=store_total))
                else:
                    setattr(cls, name, ProxyProperty(name, attr))
                self._progress[name] = attr

    cls._wrap_attributes = _wrap_attributes

    return cls


# Example usage and testing
def run_tests():
    import os
    # Setup
    db_path = 'test_nested_sqlite'
    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize the database
    with MongoDict(db_path, autocommit=True) as db:
        # Test 1: Basic nested structure
        db['subjects'] = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
        print("Initial state:")
        db.print()

        # Test 2: Update with new nested key
        db.update("subjects", {"a": {"c": 3}})
        print("\nAfter updating 'subjects.a.c':")
        db.print()

        # Test 3: Update existing nested key
        db.update("subjects", {"b": {"b": 5}})
        print("\nAfter updating 'subjects.b.b':")
        db.print()

        # Test 4: Add new top-level key
        db.update("new_key", {"x": 10, "y": 20})
        print("\nAfter adding 'new_key':")
        db.print()

        # Test 5: Update multiple levels at once
        db.update("subjects", {"a": {"d": 4}, "c": {"e": 5}})
        print("\nAfter multi-level update:")
        db.print()

        # Verify final state
        print(db["subjects"])
        assert db['subjects'] == {
            "a": {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4
            },
            "b": {
                "a": 2,
                "b": 5
            },
            "c": {
                "e": 5
            }
        }, "Final state of 'subjects' is incorrect"

        print("\nAll tests passed successfully!")

        db["subjects"] = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
        db.update("subjects", {"a": {"c": {"d": {"e": 3}}}})
        print(db["subjects"])
