"""Provides a storable dataclass with some custom dictionary operations for trackers.
Attempted thread safety but didn't work out too well. Use an external lock. 
Designed to work multiple processes.
"""
import operator
from copy import deepcopy
from pathlib import Path
from sqlitedict import SqliteDict
from typing import Any, Dict, Tuple, List
from functools import wraps


def atomic_operation(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._file_lock:
            return func(self, *args, **kwargs)

    return wrapper


from oslo_concurrency import lockutils


def atomic_operation(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        lock_name = f'progress_tracker_{func.__name__}'

        with lockutils.lock(lock_name, external=True, lock_path="lock.lock"):
            result = func(self, *args, **kwargs)
            return result

    return wrapper


class NestedSqliteDict(SqliteDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, dict):
            flat_items = list(self._flatten_dict(value, prefix=key))
            existing_keys = set(self.keys_with_prefix(key))
            keys_to_delete = existing_keys - set(k for k, _ in flat_items) - {key}

            # Batch delete
            if keys_to_delete:
                self._batch_delete(keys_to_delete)

            # Batch insert/update
            if flat_items:
                self._batch_update(flat_items)
            elif not value:
                super().__setitem__(key, value)
        else:
            keys_to_delete = set(self.keys_with_prefix(key)) - {key}
            if keys_to_delete:
                self._batch_delete(keys_to_delete)
            super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        if self._is_nested_key(key):
            return self._get_nested_dict(key)
        return super().__getitem__(key)

    def __delitem__(self, key: str):
        if self._is_nested_key(key):
            keys_to_delete = self.keys_with_prefix(key + ".")
            self._batch_delete(keys_to_delete)
        else:
            super().__delitem__(key)

    def __contains__(self, key: str) -> bool:
        return self._key_exists(key)

    def _key_exists(self, key: str) -> bool:
        if self._is_nested_key(key):
            prefix = key + "."
            return bool(self.keys_with_prefix(prefix))
        return super().__contains__(key)

    def _flatten_dict(self, d: Dict, prefix: str = '') -> List[Tuple[str, Any]]:
        items = []
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    def _is_nested_key(self, key: str) -> bool:
        prefix = key + "."
        return bool(self.keys_with_prefix(prefix))

    def _get_nested_dict(self, prefix: str) -> Dict:
        result = {}
        for key in self.keys_with_prefix(prefix + "."):
            parts = key[len(prefix) + 1:].split(".")
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = super().__getitem__(key)
        return result

    def keys_with_prefix(self, prefix: str) -> List[str]:
        query = f'SELECT key FROM "{self.tablename}" WHERE key LIKE ?'
        keys = [
            self.decode_key(key[0])
            for key in self.conn.select(query, (f"{self.encode_key(prefix)}%",))
        ]
        return keys

    def _batch_delete(self, keys: set):
        query = f'DELETE FROM "{self.tablename}" WHERE key IN ({",".join("?" for _ in keys)})'
        self.conn.execute(query, tuple(map(self.encode_key, keys)))

    def _batch_update(self, items: List[Tuple[str, Any]]):
        query = f'INSERT OR REPLACE INTO "{self.tablename}" (key, value) VALUES (?, ?)'
        self.conn.executemany(query, [(self.encode_key(k), self.encode(v)) for k, v in items])

    def print(self):
        print("=== Printing ===")
        for key, value in self.items():
            print(f'{key}: {value}')
        print("=== End Printing ===")


import operator


class InterceptedValue():
    _OPS = {
        'add': operator.add,
        'sub': operator.sub,
        'mul': operator.mul,
        'truediv': operator.truediv,
        'floordiv': operator.floordiv,
        'mod': operator.mod,
        'pow': operator.pow,
        'eq': operator.eq,
        'ne': operator.ne,
        'lt': operator.lt,
        'le': operator.le,
        'gt': operator.gt,
        'ge': operator.ge,
    }

    _ALL_OPS = [f'__{op}__' for op in _OPS] + [
        f'__r{op}__' for op in _OPS if op not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge')
    ] + [f'__i{op}__' for op in _OPS if op not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge')]

    def __init__(self,
                 name,
                 value,
                 get_callback,
                 set_callback,
                 file_lock=None,
                 store_on_init=False):
        self._name = name
        self._value = value
        self._get_callback = get_callback
        self._set_callback = set_callback
        self._file_lock = file_lock
        self._is_intercepted_value = True

        if store_on_init:
            self._set_callback(self._name, value)

    def __getattribute__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return object.__getattribute__(self, name)
        value = object.__getattribute__(self, '_value')
        if hasattr(value, name):
            return getattr(value, name)
        return object.__getattribute__(self, name)

    def __repr__(self):
        return f"InterceptedValue({self._value})"

    @classmethod
    def _create_op_method(cls, op):

        def method(self, other):
            # Get the current value using the get_callback
            self_value = self._get_callback(self._name)

            if isinstance(other, InterceptedValue):
                other = other._get_callback(other._name)

            result = cls._OPS[op](self_value, other)
            return result

        return method

    @classmethod
    def _create_rop_method(cls, op):

        def method(self, other):
            # Get the current value using the get_callback
            self_value = self._get_callback(self._name)

            result = cls._OPS[op](other, self_value)

            # Create a new InterceptedValue with the result
            return result

        return method

    @classmethod
    def _create_iop_method(cls, op):

        def method(self, other):
            # Get the current value using the get_callback
            self_value = self._get_callback(self._name)

            if isinstance(other, InterceptedValue):
                other = other._get_callback(other._name)

            result = cls._OPS[op](self_value, other)

            # Use the set_callback to update the value
            self._set_callback(self._name, result)
            self._value = result
            return self

        return method


for op in InterceptedValue._OPS:
    setattr(InterceptedValue, f'__{op}__', InterceptedValue._create_op_method(op))
    if op not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
        setattr(InterceptedValue, f'__r{op}__', InterceptedValue._create_rop_method(op))
        setattr(InterceptedValue, f'__i{op}__', InterceptedValue._create_iop_method(op))


class InterceptedStr(InterceptedValue, str):

    def __new__(cls, name, value, get_callback, set_callback, store_on_init=False, file_lock=None):
        instance = super().__new__(cls, value)
        InterceptedValue.__init__(instance,
                                  name,
                                  value,
                                  get_callback,
                                  set_callback,
                                  store_on_init=store_on_init,
                                  file_lock=file_lock)
        return instance

    def __init__(*args, **kwargs):
        pass

    def __str__(self):
        return str(self._get_callback(self._name))


class InterceptedFloat(InterceptedValue, float):

    def __new__(cls, name, value, get_callback, set_callback, store_on_init=False, file_lock=None):
        instance = super().__new__(cls, value)
        InterceptedValue.__init__(instance,
                                  name,
                                  value,
                                  get_callback,
                                  set_callback,
                                  store_on_init=store_on_init,
                                  file_lock=file_lock)
        return instance

    def __init__(*args, **kwargs):
        pass

    def __float__(self):
        return float(self._get_callback(self._name))


class InterceptedInt(InterceptedValue, int):

    def __new__(
        cls,
        name,
        value,
        get_callback,
        set_callback,
        store_on_init=False,
        file_lock=None,
    ):
        instance = super().__new__(cls, value)
        InterceptedValue.__init__(instance,
                                  name,
                                  value,
                                  get_callback,
                                  set_callback,
                                  store_on_init=store_on_init,
                                  file_lock=file_lock)
        return instance

    def __init__(*args, **kwargs):
        pass

    def __int__(self):
        return int(self._get_callback(self._name))


def wrap_value(name,
               value,
               get_callback,
               set_callback,
               store_total=False,
               store_on_init=False,
               file_lock=None):
    if isinstance(value, (InterceptedDict, InterceptedList, InterceptedValue)):
        return value
    elif isinstance(value, dict):
        return InterceptedDict(name,
                               value,
                               get_callback=get_callback,
                               set_callback=set_callback,
                               store_total=store_total,
                               file_lock=file_lock)
    elif isinstance(value, list):
        return InterceptedList(name,
                               value,
                               get_callback=get_callback,
                               set_callback=set_callback,
                               file_lock=file_lock)
    elif isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return InterceptedInt(name,
                              value,
                              get_callback,
                              set_callback,
                              store_on_init,
                              file_lock=file_lock)
    elif isinstance(value, float):
        return InterceptedFloat(name,
                                value,
                                get_callback,
                                set_callback,
                                store_on_init,
                                file_lock=file_lock)
    elif isinstance(value, str):
        return InterceptedStr(name,
                              value,
                              get_callback,
                              set_callback,
                              store_on_init,
                              file_lock=file_lock)
    return value


class InterceptedDict(dict):

    def __init__(self,
                 name,
                 value,
                 get_callback=None,
                 set_callback=None,
                 store_total=False,
                 file_lock=None,
                 **kwargs):
        self._name = name
        self._value = value
        self._file_lock = file_lock
        self._store_total = store_total
        self._get_callback = get_callback
        self._set_callback = set_callback
        super().__init__()
        self._set_self(value)
        if self._store_total:
            self["total"] = self._init_total(self._value)

    def _init_total(self, mapping: dict):
        total = 0
        for key, value in mapping.items():
            if isinstance(value, dict):
                value["total"] = self._init_total(value)
                total += value["total"]
            elif key != "total":
                total += value
        return total

    def _update_total(self, key, value):
        if isinstance(value, dict):
            value["total"] = self._init_total(value)
        if key in self:
            cur_val = self._get_callback(f"{self._name}.{key}")
            if isinstance(cur_val, dict):
                cur_val = cur_val["total"]
            if isinstance(value, dict):
                value = value["total"]
            super().__setitem__("total", self["total"] + value - cur_val)
        else:
            if isinstance(value, dict):
                value = value["total"]
            super().__setitem__("total", self["total"] + value)

    def __getitem__(self, key):
        value = self._get_callback(f"{self._name}.{key}")
        return wrap_value(f"{self._name}.{key}",
                          value,
                          self._get_callback,
                          self._set_callback,
                          store_total=self._store_total,
                          file_lock=self._file_lock)

    def __setitem__(self, key, value):
        if self._store_total and key != "total":
            self._update_total(key, value)
        super().__setitem__(key, value)
        self._set_callback(f"{self._name}.{key}", value)

    def __delitem__(self, key):
        super().__delitem__(key)
        self._set_callback(f"{self._name}.{key}", "\\delete")

    def _set_self(self, value):
        self._set_callback(self._name, value)
        for k, v in value.items():
            super().__setitem__(k, v)

    #
    def update(self, other):
        for k, v in other.items():
            if isinstance(v, dict):
                if k not in self or not isinstance(self[k], InterceptedDict):
                    self[k] = InterceptedDict(
                        f"{self._name}.{k}",
                        {},
                        self._get_callback,
                        self._set_callback,
                        self._store_total,
                    )
                self[k].update(v)
            else:
                self[k] = v
        return

    def __iadd__(self, other, *args, **kwargs):
        if set(other.keys()) - set(self.keys()):
            raise ValueError(
                f"Keys in the dictionary must match to __iadd__! stored_keys: {list(self.keys())}; added_keys: {list(other.keys())}"
            )
        for key, value in other.items():
            self[key] += value
        return self

    def __repr__(self):
        return f"InterceptedDict({super().__repr__()})"


class InterceptedList(list):

    def __init__(self,
                 name,
                 initial_value,
                 get_callback=None,
                 set_callback=None,
                 file_lock=None,
                 **kwargs):
        self._name = name
        self._file_lock = file_lock
        self._get_callback = get_callback
        self._set_callback = set_callback
        super().__init__()
        self._update_from_db()
        if initial_value and not self:  # Only extend if the list is empty
            self.extend(initial_value)

    def _update_from_db(self):
        current_value = self._get_callback(self._name)
        if isinstance(current_value, list):
            super().clear()
            super().extend(current_value)

    def __getitem__(self, index):
        self._get_callback(self._name)
        value = super().__getitem__(index)
        return wrap_value(f"{self._name}[{index}]",
                          value,
                          self._get_callback,
                          self._set_callback,
                          file_lock=self._file_lock)

    def __setitem__(self, index, value):
        self._update_from_db()
        super().__setitem__(index, value)
        self._set_callback(self._name, list(self))

    def __delitem__(self, index):
        super().__delitem__(index)
        self._set_callback(self._name, list(self))

    def append(self, value):
        self._update_from_db()
        super().append(value)
        self._set_callback(self._name, list(self))

    def extend(self, iterable):
        self._update_from_db()
        super().extend(iterable)
        self._set_callback(self._name, list(self))

    def insert(self, index, value):
        self._update_from_db()
        super().insert(index, value)
        self._set_callback(self._name, list(self))

    def pop(self, index=-1):
        self._update_from_db()
        value = super().pop(index)
        self._set_callback(self._name, list(self))
        return value

    def remove(self, value):
        self._update_from_db()
        super().remove(value)
        self._set_callback(self._name, list(self))

    def sort(self, *args, **kwargs):
        self._update_from_db()
        super().sort(*args, **kwargs)
        self._set_callback(self._name, list(self))

    def reverse(self):
        self._update_from_db()
        super().reverse()
        self._set_callback(self._name, list(self))

    def __repr__(self):
        return f"InterceptedList({self._name}, {super().__repr__()})"


class InterceptedProperty:

    def __init__(self, name, initial_value=None, store_total=False):
        self._name = name
        self._value = initial_value
        self._store_total = store_total

    def __get__(self, obj, objtype=None):
        self._value = obj._get_callback(self._name)
        if not isinstance(self._value, (InterceptedList, InterceptedValue, InterceptedDict)):
            self._value = wrap_value(self._name,
                                     self._value,
                                     obj._get_callback,
                                     obj._set_callback,
                                     store_total=self._store_total,
                                     file_lock=obj._file_lock)
        return self._value

    def __set__(self, obj, value):
        self._value = wrap_value(self._name,
                                 value,
                                 obj._get_callback,
                                 obj._set_callback,
                                 store_on_init=True,
                                 file_lock=obj._file_lock)
        if isinstance(self._value, bool):
            obj._set_callback(self._name, self._value)


def storable(cls):
    """
    A class decorator that adds persistence functionality to a class.

    Args:
        cls: The class to decorate.

    Returns:
        The decorated class.
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
        lockutils.set_defaults(lock_path=str(self._lock_file))
        self._file_lock = None

        # Load or initialize progress
        if Path(self._path.parent, f"{self._path.name}").is_file():
            self._progress = self._read()
            self._wrap_attributes()
        else:
            self._wrap_attributes()
            self._write(self._progress)

        original_init(self, *args, **kwargs)

    def _get_callback(self, key):
        # print("getting")
        with NestedSqliteDict(self._path) as db:
            ret = db[key]
            return ret

    def _set_callback(self, key, value):
        # print("setting")
        with NestedSqliteDict(self._path) as db:
            if isinstance(value, InterceptedValue):
                value = value._value
            db[key] = value
            db.commit()

    def _write(self, update_dict):
        with NestedSqliteDict(self._path) as db:
            for k, value in self._flatten_dict(update_dict):
                if isinstance(value, InterceptedValue):
                    value = value._value
                db[k] = value
            db.commit()

    def _read(self):
        with NestedSqliteDict(self._path) as db:
            ret = {}
            for key in db.keys():
                ret[key] = db[key]
            return ret

    def _db_keys(self):
        with NestedSqliteDict(self._path) as db:
            return list(db.keys())

    def _flatten_dict(self, d: Dict, prefix: str = '') -> list:
        items = []
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    def print_db(self):
        print("===== Printing =====")
        with NestedSqliteDict(self._path) as db:
            for key, value in db.items():
                print(f'{key}: {value}')
        print("===== End Printing =====")

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._wrap_attributes()

    cls.__setstate__ = __setstate__
    cls.__init__ = __init__
    cls._set_callback = _set_callback
    cls._write = _write
    cls._get_callback = _get_callback
    cls._read = _read
    cls._db_keys = _db_keys
    cls._flatten_dict = _flatten_dict
    cls.print_db = print_db

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
                    setattr(cls, name, InterceptedProperty(name, attr, store_total=store_total))
                else:
                    setattr(cls, name, InterceptedProperty(name, attr))
                self._progress[name] = attr

    cls._wrap_attributes = _wrap_attributes

    return cls


# Usage example
if __name__ == "__main__":
    # profile_storable()
    # run_performance_tests()

    @storable
    class TestClass:
        a = 1
        b = 2.2
        c = True
        _store_total = True
        d = None
        e = {'f': 3}
        f = [4, 5]

    if Path("test.sqlite").is_file():
        Path("test.sqlite").unlink()
    test = TestClass('test.sqlite')
    # test.e = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
    print(f"a: {test.a}")
    print(f"a: test.a + 1")
    test.e = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
    test.e.update({"a": {"c": 3}})
    print(f"a: test.a=1")
    test.a += 1
    print(f"a: {test.a}+=1")

    print(f"f: {test.f}")
    test.f.append(6)
    print(f"f: {test.f}")
