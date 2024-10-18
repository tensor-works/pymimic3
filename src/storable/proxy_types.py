import operator
from storable.settings import *
from storable.mongo_dict import MongoDict


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
            if READ:
                self_value = self._get_callback(self._name)
            else:
                self_value = self._value

            if isinstance(other, ProxyValue) and READ:
                other = other._get_callback(other._name)

            result = cls._OPS[op](self_value, other)
            return result

        return method

    @classmethod
    def _create_rop_method(cls, op):

        def method(self, other):
            # Get the current value using the get_callback
            if READ:
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
            if READ:
                self_value = self._get_callback(self._name)
            else:
                self_value = self._value

            if isinstance(other, ProxyValue) and READ:
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
                 db=None,
                 **kwargs):
        self._name = name
        self._value = value
        self._file_lock = file_lock
        self._store_total = store_total
        self._obj = obj
        self._get_callback = obj._get_callback
        self._set_callback = obj._set_callback
        self._del_callback = obj._del_callback
        super().__init__()
        self._set_self(value, store_on_init, db)
        if self._store_total and (store_on_init or "total" not in self):
            self.set("total", self._init_total(self, db), db=db)

    def __setitem__(self, key, value):
        wraped_value = wrap_value(self._name_encode_key(key),
                                  value,
                                  self._obj,
                                  store_total=self._store_total,
                                  store_on_init=False,
                                  file_lock=self._file_lock)
        super().__setitem__(key, wraped_value)
        self._set_callback(self._name_encode_key(key), value)
        if self._store_total and key != "total":
            self._update_totals(self._name_encode_key(key), value)

    def __delitem__(self, key):
        value = super().__getitem__(key)
        super().__delitem__(key)
        self._del_callback(self._name_encode_key(key))
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

    def __getitem__(self, key):
        if READ:
            value = self._get_callback(self._name_encode_key(key))
        else:
            # self._int_decode_key(key)
            value = super().__getitem__(key)
        return wrap_value(self._name_encode_key(key),
                          value,
                          self._obj,
                          store_total=self._store_total,
                          store_on_init=False,
                          file_lock=self._file_lock)

    def _init_total(self, mapping: dict, db=None):

        def inner(db):
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
                                          store_on_init=False,
                                          db=db)
                    cur_total = self._init_total(value)
                    value.set("total", cur_total, db=db)
                    total += cur_total
                    mapping[key] = value
                elif key != "total":
                    total += value
            return total

        if db is None:
            with MongoDict(self._obj._path) as db:
                return inner(db)
        return inner(db)

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

    def _int_decode_key(self, key):
        if key.startswith("__int__"):
            return int(key[7:])
        return key

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

    def set(self, key, value, store=True, db=None):
        super().__setitem__(key, value)
        if store:
            self._set_callback(self._name_encode_key(key), value, db)

    def _set_self(self, value, store_on_init=True, db=None):
        if store_on_init:
            self._set_callback(self._name, value, db=db)
        for k, v in value.items():
            super().__setitem__(k, v)

    def update(self, other):
        with MongoDict(self._obj._path) as db:
            self._nested_update(self, other, db)
            self._obj._update_dict(self._name, dict(self), db)
            self._update_totals(self._name, other, db)

    def _nested_update(self, current, update, db=None):
        for key, value in update.items():
            if isinstance(value, dict):
                if isinstance(current, ProxyDict):
                    cur_val = current.get(key, {}, load=False)
                else:
                    cur_val = current[key]
                if self._store_total:
                    cur_total = self._get_total_other(cur_val)

                if key not in current or not isinstance(cur_val, dict):
                    super().__setitem__(
                        key,
                        ProxyDict(f"{self._name}{NESTING_INDICATOR}{self._int_encode_keys(key)}",
                                  {},
                                  self._obj,
                                  self._store_total,
                                  self._file_lock,
                                  db=db))
                if self._store_total:
                    self_total = self.get("total", 0, load=False)
                self.get(key, load=False)._nested_update(cur_val, value, db=db)
                if self._store_total:
                    debug_total = super().__getitem__(key).get("total", load=False)
                    self.set("total", debug_total - cur_total + self_total, db=db)
            else:
                if self._store_total:
                    cur_total = super().get(key, 0)
                    self_total = self.get("total", 0, load=False)
                super().__setitem__(key, value)
                if self._store_total:
                    self.set("total", value - cur_total + self_total, db=db)
        return self

    def _get_total_other(self, other):
        if isinstance(other, ProxyDict):
            return other.get("total", 0, load=False)
        elif isinstance(other, dict):
            return other.get("total", 0)
        else:
            return other

    def _get_total(self, current=None):
        if current == None:
            current = self
        if "total" in current:
            return current.get("total", load=False)
        else:
            return sum(
                val.get("total") if isinstance(val, dict) else val for val in current.values())

    def _update_totals(self, path, value, db=None):
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
        update_locations = list(set(update_locations))
        update_locations.sort(key=lambda x: (-len(x.split(NESTING_INDICATOR)), x), reverse=True)

        for location in update_locations:
            current = self
            parts = location.split(NESTING_INDICATOR)
            for part in parts[1:]:  # Skip the first part as it's the root
                current = current[self._int_decode_key(part)]

            if isinstance(current, dict):
                old_total = current.get("total", 0)
                new_total = self._get_total(current)
                delta = new_total - old_total
                current.set("total", new_total, db=db)

                # Update parent totals incrementally
                for i in range(len(parts) - 1, 0, -1):
                    parent = self
                    for part in parts[1:i]:
                        parent = parent[part]
                    if isinstance(parent, dict):
                        parent.set("total", parent.get("total", 0) + delta, db=db)

    def clear(self):
        super().clear()
        self._set_callback(self._name, {})

    def pop(self, key, default=None):
        if key in self:
            value = super().pop(key)
            self._del_callback(self._name_encode_key(key))
            if self._store_total:
                self._update_totals(self._name_encode_key(key), value)
            return value
        return default

    def popitem(self):
        key, value = super().popitem()
        self._del_callback(self._name_encode_key(key))
        if self._store_total:
            self._update_totals(self._name_encode_key(key), value)
        return key, value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


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
        # self._update_from_db()
        if initial_value and not self:  # Only extend if the list is empty
            self.extend(initial_value)

    def _update_from_db(self):
        current_value = self._get_callback(self._name)
        if isinstance(current_value, list):
            super().clear()
            super().extend(current_value)

    def __getitem__(self, index):
        if READ:
            self._get_callback(self._name)
        value = super().__getitem__(index)
        return wrap_value(f"{self._name}[{index}]", value, self._obj, file_lock=self._file_lock)

    def __setitem__(self, index, value):
        if READ:
            self._update_from_db()
        super().__setitem__(index, value)
        self._set_callback(self._name, list(self))

    def __delitem__(self, index):
        super().__delitem__(index)
        self._set_callback(self._name, list(self))

    def append(self, value):
        if READ:
            self._update_from_db()
        super().append(value)
        self._set_callback(self._name, list(self))

    def extend(self, iterable):
        if READ:
            self._update_from_db()
        super().extend(iterable)
        self._set_callback(self._name, list(self))

    def insert(self, index, value):
        if READ:
            self._update_from_db()
        super().insert(index, value)
        self._set_callback(self._name, list(self))

    def pop(self, index=-1):
        if READ:
            self._update_from_db()
        value = super().pop(index)
        self._set_callback(self._name, list(self))
        return value

    def remove(self, value):
        if READ:
            self._update_from_db()
        super().remove(value)
        self._set_callback(self._name, list(self))

    def sort(self, *args, **kwargs):
        if READ:
            self._update_from_db()
        super().sort(*args, **kwargs)
        self._set_callback(self._name, list(self))

    def reverse(self):
        if READ:
            self._update_from_db()
        super().reverse()
        self._set_callback(self._name, list(self))

    def clear(self):
        super().clear()
        self._set_callback(self._name, [])

    def copy(self):
        return ProxyList(self._name,
                         list(self),
                         self._obj,
                         store_on_init=False,
                         file_lock=self._file_lock)

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __imul__(self, n):
        super().__imul__(n)
        self._set_callback(self._name, list(self))
        return self

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
        if READ:
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
