import re
import os
import json
import numpy as np
from pathlib import Path
from pymongo import MongoClient
from typing import Any, Dict, Tuple, List
from functools import wraps


class MongoDict:

    def __init__(self,
                 collection_name,
                 db_name=os.getenv('MONGODB_NAME', "workspace"),
                 host=os.getenv('MONGODB_HOST', None),
                 port=None,
                 autocommit=True,
                 reinit=False):
        if host is None:
            host = 'localhost'
            port = 27017
        self.client = MongoClient(host, port)

        # Use fixed database name
        self._db_name = db_name

        # Sanitize collection name using the old db name sanitization logic
        self._collection_name = self._sanitize_collection_name(collection_name)

        # Create/get the fixed database
        self.db = self.client[self._db_name]

        # TODO! consider putting this logic in the storable
        # Try restoral logic at collection level
        if not self._collection_name in self.db.list_collection_names():
            # Try to copy from original collection if progress file exists
            if Path(collection_name).exists() and collection_name:
                with open(collection_name, 'r') as f:
                    _copied_from = f.readline().strip()

                if _copied_from in self.db.list_collection_names():
                    source_coll = self.db[_copied_from]
                    docs = list(source_coll.find({}))
                    if docs:
                        self.db[self._collection_name].insert_many(docs)

            # If copying failed or no progress file, try JSON fallback
            json_path = Path(f"{collection_name}.json")
            if self._collection_name not in self.db.list_collection_names() and json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Transform the JSON structure into MongoDB documents
                    documents = []
                    for key, value in data.items():
                        documents.append({'_id': key, 'value': value})

                    if documents:
                        self.db[self._collection_name].insert_many(documents)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error loading JSON data: {e}")

        self.collection = self.db[self._collection_name]
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
    def _sanitize_collection_name(name):
        sanitized = re.sub(r'[/\\. "$*<>:|?]', '_', str(name)).strip("_")
        sanitized = 'coll_' + sanitized[-60:]
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

    def copy(self, destination: 'MongoDict'):
        """
        Copy all data from this MongoDict to another MongoDict instance.
        
        Args:
            destination (MongoDict): The destination MongoDict instance to copy data to.
        """
        # First clear the destination
        destination.clear()

        # Copy all documents from source to destination
        for doc in self.collection.find():
            key = self._decode_key(doc['_id'])
            value = self._decode_value(doc['value'])
            destination[key] = value

    def delete(self):
        """
        Delete the collection instead of the entire database
        """
        if self._collection_name in self.db.list_collection_names():
            self.db.drop_collection(self._collection_name)
            print(f"Collection '{self._collection_name}' has been deleted.")
        else:
            print(f"Collection '{self._collection_name}' does not exist.")

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
