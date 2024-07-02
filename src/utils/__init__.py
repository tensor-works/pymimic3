"""Utility file

This file implements functionalities used by other modules and accessible to 
the user

"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Union
from metrics import CustomBins, LogBins
from utils.IO import *
from pathlib import Path


def zeropad_samples(data: np.ndarray, length: int = None) -> np.ndarray:
    if length is None:
        length = max([x.shape[0] for x in data])
    ret = [
        np.concatenate([x, np.zeros((length - x.shape[0],) + x.shape[1:])],
                       axis=0,
                       dtype=np.float32) for x in data
    ]
    return np.atleast_3d(np.array(ret, dtype=np.float32))


def read_timeseries(X_df: pd.DataFrame,
                    y_df: pd.DataFrame,
                    row_only=False,
                    bining="none",
                    one_hot=False,
                    dtype=np.ndarray):
    if bining == "log":
        y_df = y_df.apply(lambda x: LogBins.get_bin_log(x, one_hot=one_hot))
    elif bining == "custom":
        y_df = y_df.apply(lambda x: CustomBins.get_bin_custom(x, one_hot=one_hot))
    else:
        y_df = y_df.astype(np.float32)

    if row_only:
        Xs = [
            X_df.loc[timestamp].values if dtype in [np.ndarray, np.array] else X_df.loc[timestamp]
            for timestamp in y_df.index
        ]
    else:
        Xs = [
            X_df.loc[:timestamp].values if dtype in [np.ndarray, np.array] else X_df.loc[:timestamp]
            for timestamp in y_df.index
        ]

    ys = y_df.squeeze(axis=1).values
    ts = y_df.index.tolist()

    return Xs, ys, ts


def is_numerical(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame contains only numerical data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    bool
        True if the DataFrame is numerical, False otherwise.
    """
    # This is the worst implementation but what works works
    try:
        df.astype(float)
        return True
    except:
        pass
    if (df.dtypes == object).any() or\
       (df.dtypes == "category").any() or\
       (df.dtypes == "datetime64[ns]").any():
        return False
    return True


def to_snake_case(string):
    """
    Convert a string to snake case.

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    str
        The converted string in snake case.
    """
    # Replace spaces and hyphens with underscores
    string = re.sub(r'[\s-]+', '_', string)

    # Replace camel case and Pascal case with underscores
    string = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', string)

    # Convert the string to lowercase
    string = string.lower()

    return string


def is_colwise_numerical(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Check if each column in a DataFrame is numerical.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    dict
        A dictionary with column names as keys and boolean values indicating if the column is numerical.
    """
    return {col: is_numerical(df[[col]]) for col in df.columns}


def is_allnan(data: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """
    Check if all elements in the data are NaN.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray]
        The data to check.

    Returns
    -------
    bool
        True if all elements are NaN, False otherwise.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.isna().all().all()
    elif isinstance(data, np.ndarray):
        return np.isnan(data).all()
    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or numpy array.")


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


def is_iterable(obj):
    """
    Check if an object is iterable.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if the object is iterable, False otherwise.
    """
    return hasattr(obj, '__iter__')


def make_prediction_vector(model, generator, batches=20, bin_averages=None):
    """_summary_
    """
    # TODO! fix bin averages instead
    Xs = list()
    ys = list()

    for _ in range(batches):
        X, y = generator.next()
        Xs.append(X)
        ys.append(y)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate([model.predict(X, verbose=0) for X in Xs])

    if bin_averages:
        # TODO! shape mismatch betwenn prediction vector length and length of bin averages
        y_pred = np.array([
            bin_averages[int(label)]
            if label < len(bin_averages) else bin_averages[len(bin_averages) - 1]
            for label in np.argmax(y_pred, axis=1)
        ]).reshape(1, -1)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        y_true = np.array([
            bin_averages[int(label)]
            if label < len(bin_averages) else bin_averages[len(bin_averages) - 1]
            for label in y_true
        ]).reshape(1, -1)

    # TODO! this should be y_pred y_true
    return y_pred, y_true


def count_csv_size(file_path: Path):

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        file_length = sum(bl.count("\n") for bl in blocks(f))

    return file_length - 1


class NoopLock:

    def __enter__(self):
        # Do nothing
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Do nothing
        pass


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
