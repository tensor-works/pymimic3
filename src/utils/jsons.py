import numpy as np
import json
from utils.IO import info_io


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dict_subset(dictionary: dict, keys: list):
    """
    Get a subset of a dictionary.

    Parameters
    ----------
    dictionary : dict
        The original dictionary.
    keys : list
        The keys to extract from the dictionary.

    Returns
    -------
    dict
        A dictionary containing only the specified keys.
    """
    return {k: dictionary[k] for k in keys if k in dictionary}


def write_json(json_path, json_data):
    """
    Write data to a JSON file.

    Parameters
    ----------
    json_path : Path
        The path to the JSON file.
    json_data : dict
        The data to write to the JSON file.
    """
    try:
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4, cls=NpEncoder)
    except KeyboardInterrupt:
        info_io("Finishing JSON operation before interupt.")
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4, cls=NpEncoder)

    return


def load_json(json_path):
    """
    Load data from a JSON file.

    Parameters
    ----------
    json_path : Path
        The path to the JSON file.

    Returns
    -------
    dict
        The loaded JSON data.
    """
    if not json_path.is_file():
        return {}

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    return json_data


def update_json(json_path, items: dict):
    """
    Update a JSON file with new items.

    Parameters
    ----------
    json_path : Path
        The path to the JSON file.
    items : dict
        The items to update the JSON file with.

    Returns
    -------
    dict
        The updated JSON data.
    """
    if not json_path.parent.is_dir():
        json_path.parent.mkdir(parents=True, exist_ok=True)
    if not json_path.is_file():
        with open(json_path, 'w+') as file:
            json.dump({}, file, cls=NpEncoder)

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    json_data.update(items)
    try:
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4, cls=NpEncoder)
    except KeyboardInterrupt:
        info_io("Finishing JSON operation before interupt.")
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4, cls=NpEncoder)

    return json_data
