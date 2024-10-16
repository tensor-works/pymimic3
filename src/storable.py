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

NESTING_INDICATOR = "^"

from oslo_concurrency import lockutils


def atomic_operation(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        lock_name = f'progress_tracker_{func.__name__}'

        with lockutils.lock(lock_name, external=True, lock_path="lock.lock"):
            result = func(self, *args, **kwargs)
            return result

    return wrapper


import ast


class NestedSqliteDict(SqliteDict):
    """
    A subclass of SqliteDict that supports nested dictionary operations.

    This class provides methods for working with nested dictionaries in a SQLite database,
    including flattening and reconstructing nested structures.

    Methods
    -------
    __setitem__(key: str, value: Any)
        Set an item in the nested dictionary structure.
    __getitem__(key: str) -> Any
        Get an item from the nested dictionary structure.
    __delitem__(key: str)
        Delete an item from the nested dictionary structure.
    __contains__(key: str) -> bool
        Check if a key exists in the nested dictionary structure.
    keys_with_prefix(prefix: str) -> List[str]
        Get all keys with a given prefix.
    print()
        Print the contents of the database.
    
    Examples
    --------
    >>> db = NestedSqliteDict('example.sqlite', autocommit=True)
    >>> db['user.profile.name'] = 'John Doe'
    >>> db['user.profile.age'] = 30
    >>> db['user.settings.theme'] = 'dark'
    >>> print(db['user.profile'])
    {'name': 'John Doe', 'age': 30}
    >>> print(db['user'])
    {'profile': {'name': 'John Doe', 'age': 30}, 'settings': {'theme': 'dark'}}
    >>> 'user.profile.name' in db
    True
    >>> db.keys_with_prefix('user.profile')
    ['user.profile.name', 'user.profile.age']
    >>> del db['user.settings.theme']
    >>> db.print()
    >>> db.close()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        encoded_key = self._encode_key(key)
        value = self._encode_value(value)
        if isinstance(value, dict):
            flat_items = list(self._flatten_dict(value, prefix=encoded_key))
            existing_keys = set(self.keys_with_prefix(encoded_key))
            keys_to_delete = existing_keys - set(k for k, _ in flat_items)  # - {encoded_key}

            if keys_to_delete:
                self._batch_delete(keys_to_delete)

            if flat_items:
                self._batch_update(flat_items)
            elif not value:
                super().__setitem__(encoded_key, dict(value))
        else:
            keys_to_delete = set(self.keys_with_prefix(encoded_key)) - {encoded_key}
            if keys_to_delete:
                self._batch_delete(keys_to_delete)
            super().__setitem__(encoded_key, value)

    def _encode_value(self, value):
        if isinstance(value, ProxyValue):
            return self._encode_value(value._value)
        elif isinstance(value, dict):
            return {self._encode_key(k): self._encode_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._encode_value(v) for v in value]
        return value

    def __getitem__(self, key):
        encoded_key = self._encode_key(key)
        if self._is_nested_key(encoded_key):
            return self._get_nested_dict(encoded_key)
        try:
            return super().__getitem__(encoded_key)
        except KeyError as e:
            self.print()
            raise KeyError(f"Key {key} not found in the database {list(self.keys())}.")

    def __delitem__(self, key):
        encoded_key = self._encode_key(key)
        if self._is_nested_key(encoded_key):
            keys_to_delete = self.keys_with_prefix(encoded_key + ".")
            self._batch_delete(keys_to_delete)
        else:
            super().__delitem__(encoded_key)

    def __contains__(self, key):
        encoded_key = self._encode_key(key)
        return self.key_exists(encoded_key)

    def key_exists(self, key):
        encoded_key = self._encode_key(key)
        if self._is_nested_key(encoded_key):
            prefix = encoded_key + "."
            return bool(self.keys_with_prefix(prefix))
        return super().__contains__(encoded_key)

    def _flatten_dict(self, d, prefix=''):
        items = []
        for k, v in d.items():
            encoded_k = self._encode_key(k)
            new_key = f"{prefix}{NESTING_INDICATOR}{encoded_k}" if prefix else encoded_k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    def _is_nested_key(self, key):
        prefix = key + NESTING_INDICATOR
        return bool(self.keys_with_prefix(prefix))

    def _get_nested_dict(self, prefix):
        result = {}
        key_with_prefix = self.keys_with_prefix(prefix + NESTING_INDICATOR)
        for key in key_with_prefix:
            parts = key[len(prefix) + 1:].split(NESTING_INDICATOR)
            current = result
            for part in parts[:-1]:
                decoded_part = self._decode_key(part)
                current = current.setdefault(decoded_part, {})
            decoded_last_part = self._decode_key(parts[-1])
            current[decoded_last_part] = super().__getitem__(key)
        return result

    def keys_with_prefix(self, prefix):
        encoded_prefix = self._encode_key(prefix)
        query = f'SELECT key FROM "{self.tablename}" WHERE key LIKE ?'
        keys = [
            self.decode_key(key[0])
            for key in self.conn.select(query, (f"{self.encode_key(encoded_prefix)}%",))
        ]
        return keys

    def _batch_delete(self, keys):
        encoded_keys = [self._encode_key(key) for key in keys]
        query = f'DELETE FROM "{self.tablename}" WHERE key IN ({",".join("?" for _ in encoded_keys)})'
        self.conn.execute(query, tuple(map(self.encode_key, encoded_keys)))

    def _batch_update(self, items):
        encoded_items = [(self._encode_key(k), v) for k, v in items]
        query = f'INSERT OR REPLACE INTO "{self.tablename}" (key, value) VALUES (?, ?)'
        self.conn.executemany(query,
                              [(self.encode_key(k), self.encode(v)) for k, v in encoded_items])

    def _encode_key(self, key):
        if isinstance(key, int):
            return f"__int__{str(key)}"
        return str(key)

    def _decode_key(self, key):
        if key.startswith("__int__"):
            return int(key[7:])
        return key

    def keys(self):
        return [self._decode_key(key) for key in super().keys()]

    def items(self):
        return [(self._decode_key(k), v) for k, v in super().items()]

    def print(self):
        print("=== Printing ===")
        for key, value in self.items():
            print(f'{key}: {value}')
        print("=== End Printing ===")

    def update(self, attribute, update_dict):
        base_key = self._encode_key(attribute)

        def update_recursive(current_key, current_value):
            if isinstance(current_value, dict):
                for k, v in current_value.items():
                    new_key = f"{current_key}{NESTING_INDICATOR}{self._encode_key(k)}"
                    update_recursive(new_key, v)
            else:
                self[current_key] = current_value

        update_recursive(base_key, update_dict)

    def _nested_update(self, existing, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in existing and isinstance(existing[k], dict):
                self._nested_update(existing[k], v)
            else:
                existing[k] = v


import operator


class ProxyValue():
    """
    A class that intercepts operations on values and provides callbacks for get and set operations.

    This class is used to create proxy objects for primitive types (int, float, str)
    that can trigger callbacks when accessed or modified and implement inplace modification methods.
    This was designed to work with the storable class, and enable thread safety.

    Methods
    -------
    Various magic methods are implemented to intercept different operations.
    """

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

    def __init__(self, name, value, obj, file_lock=None, store_on_init=False):
        self._name = name
        self._value = value
        self._obj = obj
        self._get_callback = obj._get_callback
        self._set_callback = obj._set_callback
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
        return f"{self._value}"

    @classmethod
    def _create_op_method(cls, op):

        def method(self, other):
            # Get the current value using the get_callback
            if False:
                self_value = self._get_callback(self._name)
            else:
                self_value = self._value

            if isinstance(other, ProxyValue) and False:
                other = other._get_callback(other._name)

            result = cls._OPS[op](self_value, other)
            return result

        return method

    @classmethod
    def _create_rop_method(cls, op):

        def method(self, other):
            # Get the current value using the get_callback
            if False:
                self_value = self._get_callback(self._name)
            else:
                self_value = self._value

            result = cls._OPS[op](other, self_value)

            # Create a new InterceptedValue with the result
            return result

        return method

    @classmethod
    def _create_iop_method(cls, op):

        def method(self, other):
            # Get the current value using the get_callback
            if False:
                self_value = self._get_callback(self._name)
            else:
                self_value = self._value

            if isinstance(other, ProxyValue) and False:
                other = other._get_callback(other._name)

            result = cls._OPS[op](self_value, other)

            # Use the set_callback to update the value
            self._set_callback(self._name, result)
            self._value = result
            return self

        return method


for op in ProxyValue._OPS:
    setattr(ProxyValue, f'__{op}__', ProxyValue._create_op_method(op))
    if op not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
        setattr(ProxyValue, f'__r{op}__', ProxyValue._create_rop_method(op))
        setattr(ProxyValue, f'__i{op}__', ProxyValue._create_iop_method(op))


class ProxyStr(ProxyValue, str):

    def __new__(cls, name, value, obj, store_on_init=False, file_lock=None):
        instance = super().__new__(cls, value)
        ProxyValue.__init__(instance,
                            name,
                            value,
                            obj,
                            store_on_init=store_on_init,
                            file_lock=file_lock)
        return instance

    def __init__(*args, **kwargs):
        pass

    def __str__(self):
        return str(self._get_callback(self._name))


class ProxyFloat(ProxyValue, float):

    def __new__(cls, name, value, obj, store_on_init=False, file_lock=None):
        instance = super().__new__(cls, value)
        ProxyValue.__init__(instance,
                            name,
                            value,
                            obj,
                            store_on_init=store_on_init,
                            file_lock=file_lock)
        return instance

    def __init__(*args, **kwargs):
        pass

    def __float__(self):
        return float(self._get_callback(self._name))


class ProxyInt(ProxyValue, int):

    def __new__(
        cls,
        name,
        value,
        obj,
        store_on_init=False,
        file_lock=None,
    ):
        instance = super().__new__(cls, value)
        ProxyValue.__init__(instance,
                            name,
                            value,
                            obj,
                            store_on_init=store_on_init,
                            file_lock=file_lock)
        return instance

    def __init__(*args, **kwargs):
        pass

    def __int__(self):
        return int(self._get_callback(self._name))


def wrap_value(name, value, obj, store_total=False, store_on_init=False, file_lock=None):
    if isinstance(value, (ProxyDict, ProxyList, ProxyValue)):
        return value
    elif isinstance(value, dict):
        return ProxyDict(name,
                         value,
                         obj,
                         store_total=store_total,
                         store_on_init=store_on_init,
                         file_lock=file_lock)
    elif isinstance(value, list):
        return ProxyList(name, value, obj, store_on_init=store_on_init, file_lock=file_lock)
    elif isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return ProxyInt(name, value, obj, store_on_init, file_lock=file_lock)
    elif isinstance(value, float):
        return ProxyFloat(name, value, obj, store_on_init, file_lock=file_lock)
    elif isinstance(value, str):
        return ProxyStr(name, value, obj, store_on_init, file_lock=file_lock)
    return value


class ProxyDict(dict):
    """
    A dictionary subclass that intercepts get and set operations.

    This class is used to create a proxy dictionary that can trigger callbacks
    when items are accessed or modified.

    Methods
    -------
    __getitem__(key)
        Get an item from the dictionary, wrapping it if necessary.
    __setitem__(key, value)
        Set an item in the dictionary, updating totals if necessary.
    __delitem__(key)
        Delete an item from the dictionary.
    update(other)
        Update the dictionary with another dictionary.
    __iadd__(other)
        Implement the += operation for the dictionary.
    """

    def __init__(self,
                 name,
                 value,
                 obj,
                 store_total=False,
                 file_lock=None,
                 load_on_init=True,
                 store_on_init=True,
                 **kwargs):
        self._name = name
        self._value = value
        self._file_lock = file_lock
        self._store_total = store_total
        self._obj = obj
        self._get_callback = obj._get_callback
        self._set_callback = obj._set_callback
        super().__init__()
        self._set_self(value, store_on_init)
        if self._store_total and store_on_init:
            self["total"] = self._init_total(self)
        return

    def _init_total(self, mapping: dict):
        total = 0
        keys = list(mapping.keys())
        for key in keys:
            if isinstance(mapping, ProxyDict):
                value = mapping.get(key, load=False)
            else:
                value = mapping[key]

            if isinstance(value, dict):
                if not isinstance(value, ProxyDict):
                    value = ProxyDict(self._name_encode_key(key),
                                      value,
                                      self._obj,
                                      self._store_total,
                                      self._file_lock,
                                      store_on_init=False)
                cur_total = self._init_total(value)
                value["total"] = cur_total
                total += cur_total
                mapping[key] = value
            elif key != "total":
                total += value
        return total

    def _update_from_db(self):
        db_state = self._get_callback(self._name)
        if not isinstance(db_state, dict):
            raise ValueError(f"Database state for {self._name} is not a dictionary")
        super().clear()

        for key, value in db_state.items():
            if isinstance(value, dict):
                super().__setitem__(
                    key,
                    ProxyDict(self._name_encode_key(key),
                              value,
                              self._obj,
                              self._store_total,
                              self._file_lock,
                              store_on_init=False))
            else:
                super().__setitem__(key, value)
        return

    def _name_encode_key(self, key):
        return f"{self._name}{NESTING_INDICATOR}{self._int_encode_keys(key)}"

    def _int_encode_keys(self, key):
        if isinstance(key, int):
            return f"__int__{key}"
        return str(key)

    def __getitem__(self, key):
        value = self._get_callback(self._name_encode_key(key))
        return wrap_value(self._name_encode_key(key),
                          value,
                          self._obj,
                          store_total=self._store_total,
                          store_on_init=False,
                          file_lock=self._file_lock)

    def get(self, key, default=None, load=True):
        if key in self and not load:
            value = super().__getitem__(key)
            super().__setitem__(
                key,
                wrap_value(self._name_encode_key(key),
                           value,
                           self._obj,
                           store_total=self._store_total,
                           store_on_init=False,
                           file_lock=self._file_lock))
            return super().__getitem__(key)
        if key in self:
            return self[key]
        return default

    def set(self, key, value, store=True):
        super().__setitem__(key, value)
        if store:
            self._set_callback(self._name_encode_key(key), value)

    def _set_self(self, value, store_on_init=True):
        if store_on_init:
            self._set_callback(self._name, value)
        for k, v in value.items():
            super().__setitem__(k, v)

    def update(self, other):
        self._nested_update(self, other)
        self._obj._update_dict(self._name, dict(self))
        # self._update_totals(self._name, other)

    def _nested_update(self, current, update):
        for key, value in update.items():
            print(f"Name: {self._name}")
            if isinstance(value, dict):
                if isinstance(current, ProxyDict):
                    cur_val = current.get(key, {}, load=False)
                    if self._store_total:
                        cur_total = self._get_total_other(cur_val)
                else:
                    cur_val = current[key]
                    if self._store_total:
                        cur_total = self._get_total_other(cur_val)

                if key not in current or not isinstance(cur_val, dict):
                    super().__setitem__(
                        key,
                        ProxyDict(f"{self._name}{NESTING_INDICATOR}{self._int_encode_keys(key)}",
                                  {}, self._obj, self._store_total, self._file_lock))
                if self._store_total:
                    self_total = self.get("total", 0, load=False)
                self.get(key, load=False)._nested_update(cur_val, value)
                if self._store_total:
                    debug_total = super().__getitem__(key).get("total", load=False)
                    self["total"] = debug_total - cur_total + self_total
            else:
                if self._store_total:
                    cur_total = super().get(key, 0)
                    self_total = self.get("total", 0, load=False)
                super().__setitem__(key, value)
                if self._store_total:
                    self["total"] = value - cur_total + self_total
        return self

    def _get_total_other(self, other):
        if isinstance(other, ProxyDict):
            return other.get("total", 0, load=False)
        elif isinstance(other, dict):
            return other.get("total", 0)
        else:
            return other

    def _get_total(self):
        if "total" in self:
            return self.get("total", load=False)
        else:
            return sum(val.get("total") if isinstance(val, dict) else val for val in self.values())

    def _update_totals(self, path, value):
        if not self._store_total:
            return

        update_locations = list()

        def find_locations(values, current_path):
            if isinstance(values, dict):
                for key, val in values.items():
                    if isinstance(val, dict):
                        find_locations(
                            val, f"{current_path}{NESTING_INDICATOR}{self._int_encode_keys(key)}")
                    elif key != "total":
                        update_locations.append(current_path)
            else:
                update_locations.append(current_path)

        find_locations(value, path)

        # Sort update_locations from deepest to shallowest
        update_locations.sort(key=lambda x: (-len(x.split(NESTING_INDICATOR)), x), reverse=True)

        for location in update_locations:
            current = self
            parts = location.split(NESTING_INDICATOR)
            for part in parts[1:]:  # Skip the first part as it's the root
                current = current[part]

            if isinstance(current, dict):
                old_total = current.get("total", 0)
                new_total = sum(v for k, v in current.items() if k != "total")
                delta = new_total - old_total
                current["total"] = new_total
                self._set_callback(f"{location}{NESTING_INDICATOR}total", new_total)

                # Update parent totals incrementally
                for i in range(len(parts) - 1, 0, -1):
                    parent = self
                    for part in parts[1:i]:
                        parent = parent[part]
                    if isinstance(parent, dict):
                        parent["total"] = parent.get("total", 0) + delta
                        self._set_callback(f"{NESTING_INDICATOR}".join(parts[:i] + ["total"]),
                                           parent["total"])

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._set_callback(self._name_encode_key(key), value)
        if self._store_total and key != "total":
            self._update_totals(self._name_encode_key(key), value)

    def __delitem__(self, key):
        value = super().__getitem__(key)
        super().__delitem__(key)
        self._set_callback(self._name_encode_key(key), "\\delete")
        if self._store_total:
            self._update_totals(self._name_encode_key(key), value)

    def __iadd__(self, other, *args, **kwargs):
        if set(other.keys()) - set(self.keys()):
            raise ValueError(
                f"Keys in the dictionary must match to __iadd__! stored_keys: {list(self.keys())}; added_keys: {list(other.keys())}"
            )
        for key, value in other.items():
            self[key] = value + self[key]
        return self

    def __repr__(self):
        return f"{super().__repr__()}"


class ProxyList(list):
    """
    A list subclass that intercepts get and set operations.

    This class is used to create a proxy list that can trigger callbacks
    when items are accessed or modified.

    Methods
    -------
    Various list methods are overridden to provide interception functionality.
    """

    def __init__(self, name, initial_value, obj, store_on_init=True, file_lock=None, **kwargs):
        self._name = name
        self._file_lock = file_lock
        self._obj = obj
        self._get_callback = obj._get_callback
        self._set_callback = obj._set_callback
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
        return wrap_value(f"{self._name}[{index}]", value, self._obj, file_lock=self._file_lock)

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
        return f"{super().__repr__()}"


class ProxyProperty:
    """
    A descriptor class for intercepting attribute access and modification.

    This class is used to create property-like objects that can trigger
    callbacks when accessed or modified.

    Methods
    -------
    __get__(obj, objtype=None)
        Get the value of the property.
    __set__(obj, value)
        Set the value of the property.
    """

    def __init__(self, name, initial_value=None, store_total=False):
        self._name = name
        self._value = initial_value
        self._store_total = store_total

    def __get__(self, obj, objtype=None):
        if not isinstance(self._value, (ProxyList, ProxyValue, ProxyDict)) and obj is not None:
            self._value = wrap_value(self._name,
                                     self._value,
                                     obj,
                                     store_total=self._store_total,
                                     file_lock=obj._file_lock)

        # Update to newest version on get
        if False:
            if isinstance(self._value, ProxyValue):
                self._value._value = self._value._get_callback(self._name)
            elif isinstance(self._value, (ProxyList, ProxyDict)):
                self._value._update_from_db()
            else:
                self._value = obj._get_callback(self._name)
        return self._value

    def __set__(self, obj, value):
        if isinstance(value, (ProxyDict, ProxyList)):
            self._value = value
            self._value._set_callback(self._name, value)
        elif isinstance(value, ProxyValue):
            self._value = value
            self._value._set_callback(self._name, value._value)
        elif isinstance(value, bool):
            self._value = value
            obj._set_callback(self._name, value)
        elif isinstance(value, (dict, list, str, float, int)) and obj is not None:
            self._value = wrap_value(self._name,
                                     value,
                                     obj,
                                     store_total=self._store_total,
                                     store_on_init=True,
                                     file_lock=obj._file_lock)
        else:
            self._value = value
            if obj is not None:
                obj._set_callback(self._name, value)


import sqlitedict


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
        with NestedSqliteDict(self._path) as db:
            ret = db[key]
            return ret

    def _set_callback(self, key, value):
        with NestedSqliteDict(self._path) as db:
            if isinstance(value, ProxyValue):
                value = value._value
            db[key] = value
            db.commit()

    def _update_dict(self, attribute, updates):
        with NestedSqliteDict(self._path) as db:
            db.update(attribute, updates)
            db.commit()

    def _set_callback_bulk(self, updates):
        with NestedSqliteDict(self._path) as db:
            for key, value in updates.items():
                if isinstance(value, ProxyValue):
                    value = value._value
                db[key] = value
            db.commit()

    def _write(self, update_dict):
        with NestedSqliteDict(self._path) as db:
            for k, value in update_dict.items():
                if isinstance(value, ProxyValue):
                    value = value._value
                db[k] = value
            db.commit()

    def _key_exists(self, key):
        with NestedSqliteDict(self._path) as db:
            return db.key_exists(key)

    def _read(self):
        with NestedSqliteDict(self._path) as db:
            ret = {}
            _progress_keys = self._progress_keys()
            for key in _progress_keys:
                ret[key] = db[key]
            return ret

    def _db_keys(self):
        with NestedSqliteDict(self._path) as db:
            return list(db.keys())

    def _progress_keys(self):
        return list(set([key.split("^")[0] for key in self._db_keys()]))

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
    cls._progress_keys = _progress_keys
    cls._update_dict = _update_dict
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
    db_path = 'test_nested_sqlite.db'
    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize the database
    with NestedSqliteDict(db_path, autocommit=True) as db:
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


# Usage example
if __name__ == "__main__":
    run_tests()
    # profile_storable()
    # run_performance_tests()

    @storable
    class TestClass:
        # a = 1
        # b = 2.2
        # c = True
        # _store_total = True
        # d = None
        e = {"a": 0, "b": 0}
        # f = [4, 5]

    if Path("test.sqlite").is_file():
        Path("test.sqlite").unlink()
    test = TestClass('test.sqlite')
    test.e += {"a": 1, "b": 2}
    with NestedSqliteDict('test.sqlite') as db:
        db.print()
    print(test.e)

    exit()
    del test
    test = TestClass('test.sqlite')
    print(test.e)

    test.e[1] = 1
    print(f"f: {test.f}")
    # test.e = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
    print(f"a: {test.a}")
    print(f"a: test.a + 1")
    test.e = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
    test.e.update({"a": {"c": 3}})
    print(test.e)
    print(f"a: test.a=1")
    test.a += 1
    print(f"a: {test.a}+=1")

    print(f"f: {test.f}")
    test.f.append(6)
    print(f"f: {test.f}")
    test.n = {"a.b.c": 1}
    print(f"n: {test.n}")

    @storable
    class TestClass:
        a = 1
        b = 2.2
        c = True
        _store_total = True
        d = None
        e = {'a': 3}
        f = [4, 5]
        subjects = {"a": 0, "b": 0}

    if Path("atest.sqlite").is_file():
        Path("atest.sqlite").unlink()
    a = TestClass('atest.sqlite')
    # print(a.e)
    a.e += {"a": 4}
    print(a.e)
    a.e.update({"a": {"b": 5}})
    print(a.e)
    a.e.update({"a": {"c": 5}})
    print(a.e)
    a.e.update({"b": {"o": 5}})
    with NestedSqliteDict('atest.sqlite') as db:
        db.print()
    print(a.e)
    del a

    a = TestClass('atest.sqlite')
    print(a.e)
    a.f = None
    with NestedSqliteDict('atest.sqlite') as db:
        db.print()
    print(a.f)

    a.subjects = {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 4}}
    print(a.subjects)
