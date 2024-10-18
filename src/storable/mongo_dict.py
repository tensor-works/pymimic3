import re
import json
import numpy as np
from pymongo import MongoClient
from typing import Any, Dict, Tuple, List
from functools import wraps


class MongoDict:

    def __init__(self,
                 _db_name,
                 collection_name='default_collection',
                 host='localhost',
                 port=27017,
                 autocommit=True,
                 reinit=False):
        self.client = MongoClient(host, port)
        self._db_name = self._sanitize_db_name(_db_name)
        self.db = self.client[self._db_name]
        self.collection = self.db[collection_name]
        self.autocommit = autocommit
        if reinit:
            self.delete()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def __contains__(self, key):
        encoded_key = self._encode_key(key)
        return self.collection.count_documents({'_id': encoded_key}) > 0

    def __getitem__(self, key):
        encoded_key = self._encode_key(key)
        result = self.collection.find_one({'_id': encoded_key})
        if result:
            return self._decode_value(result['value'])
        raise KeyError(key)

    def _encode_key(self, key):
        if isinstance(key, int):
            return f"__int__{str(key)}"
        return str(key)

    @staticmethod
    def _sanitize_db_name(name):
        sanitized = re.sub(r'[/\\. "$*<>:|?]', '_', str(name)).strip("_")
        sanitized = 'db_' + sanitized[-60:]
        # Truncate to 63 bytes (MongoDB's limit)
        return sanitized

    def _decode_key(self, key):
        if key.startswith("__int__"):
            return int(key[7:])
        return key

    def _encode_value(self, value):
        if isinstance(value, dict):
            return {self._encode_key(k): self._encode_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._encode_value(item) for item in value]
        elif isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _decode_value(self, value):
        if isinstance(value, dict):
            return {self._decode_key(k): self._decode_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._decode_value(item) for item in value]
        return value

    def __setitem__(self, key, value):
        encoded_key = self._encode_key(key)
        encoded_value = self._encode_value(value)
        self.collection.update_one({'_id': encoded_key}, {'$set': {
            'value': encoded_value
        }},
                                   upsert=True)

    def update(self, key, value):
        encoded_key = self._encode_key(key)
        if encoded_key not in self:
            self[key] = {}

        def update_nested(existing, new_data):
            for k, v in new_data.items():
                if isinstance(v, dict) and k in existing and isinstance(existing[k], dict):
                    update_nested(existing[k], v)
                else:
                    existing[k] = v

        existing_data = self[key]
        update_nested(existing_data, value)
        self[key] = existing_data

    def delete(self):
        """
        Delete the entire database if it exists.
        """
        if self._db_name in self.client.list_database_names():
            self.client.drop_database(self._db_name)
            print(f"Database '{self._db_name}' has been deleted.")
        else:
            print(f"Database '{self._db_name}' does not exist.")

    def print(self):
        for doc in self.collection.find():
            decoded_key = self._decode_key(doc['_id'])
            print(f"{decoded_key}: {json.dumps(doc['value'], indent=2)}")

    def clear(self):
        self.collection.delete_many({})

    def delete_nested(self, key_path):
        keys = key_path.split('.')
        top_level_key = self._encode_key(keys[0])

        # Retrieve the document
        doc = self.collection.find_one({'_id': top_level_key})
        if not doc:
            return False

        # Navigate to the nested key
        current = doc['value']
        parent_stack = []
        for key in keys[1:-1]:
            if isinstance(current, dict) and key in current:
                parent_stack.append((current, key))
                current = current[key]
            else:
                return False

        # Delete the key
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]
        else:
            return False

        # Clean up empty parent dictionaries
        while parent_stack:
            parent, key = parent_stack.pop()
            if not parent[key]:
                del parent[key]
            else:
                break

        # If the top-level dictionary is now empty, delete the entire document
        if not doc['value']:
            self.collection.delete_one({'_id': top_level_key})
        else:
            # Update the document in the database
            self.collection.update_one({'_id': top_level_key}, {'$set': doc})

        return True

    def set_nested(self, key_path, value):
        if not key_path:
            raise ValueError("Key path cannot be empty")

        keys = key_path.split('.')
        top_level_key = self._encode_key(keys[0])

        if len(keys) == 1:
            self[self._decode_key(top_level_key)] = value
            return

        # Retrieve existing document or create a new one
        doc = self.collection.find_one({'_id': top_level_key}) or {
            '_id': top_level_key,
            'value': {}
        }

        # Navigate through the nested structure
        current = doc['value']
        for i, key in enumerate(keys[1:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Set the value at the final level
        if isinstance(current, dict):
            current[keys[-1]] = self._encode_value(value)
        else:
            # If the current value is not a dict, we need to replace it entirely
            parent = doc['value']
            for key in keys[1:-2]:
                parent = parent[key]
            parent[keys[-2]] = {keys[-1]: self._encode_value(value)}

        # Update the document in the database
        self.collection.update_one({'_id': top_level_key}, {'$set': doc}, upsert=True)

    def get_nested(self, key_path):
        keys = key_path.split('.')
        top_level_key = self._encode_key(keys[0])

        if len(keys) == 1:
            return self[self._decode_key(top_level_key)]

        field_path = 'value.' + '.'.join(keys[1:])
        result = self.collection.find_one({'_id': top_level_key}, {field_path: 1})

        if not result:
            raise KeyError(f"Key '{self._decode_key(top_level_key)}' not found in the database")

        value = result
        for key in keys[1:]:
            if isinstance(value, dict) and 'value' in value:
                value = value['value']
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Nested key '{key}' not found in the path '{key_path}'")

        return value

    def items(self):
        for doc in self.collection.find():
            decoded_key = self._decode_key(doc['_id'])
            yield (decoded_key, doc['value'])

    def to_dict(self):
        """
        Read the entire NestedMongoDict into a Python dictionary.
        """
        result = {}
        for doc in self.collection.find():
            key = self._decode_key(doc['_id'])
            value = self._decode_value(doc['value'])
            result[key] = value
        return result

    def print_all(self):
        """
        Print all key-value pairs in the NestedMongoDict.
        """
        for key, value in self.to_dict().items():
            print(f"{key}: {json.dumps(value, indent=2)}")
