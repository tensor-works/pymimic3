"""Utility file

This file implements functionalities used by other modules and accessible to 
the user

Todo:
    - Use a settings.json
    - fix the progressbar bullshit

"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Union
from multipledispatch import dispatch

from utils.IO import *
from pathlib import Path


@dispatch(dict)
def get_sample_size(X):
    """
    """
    n_samples = 0
    for data in X.values():
        if isinstance(data, dict):
            n_samples += get_sample_size(data)
        else:
            n_samples += len(data)

    return n_samples


@dispatch(list)
def get_sample_size(X):
    """
    """
    n_samples = 0
    for subject in X:
        if isinstance(X, dict):
            for stay in X:
                n_samples += len(stay)
        else:
            n_samples += len(subject)

    return n_samples


def is_numerical(df: pd.DataFrame) -> bool:
    # Check if the DataFrame is numerical
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
    # Replace spaces and hyphens with underscores
    string = re.sub(r'[\s-]+', '_', string)

    # Replace camel case and Pascal case with underscores
    string = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', string)

    # Convert the string to lowercase
    string = string.lower()

    return string


def is_colwise_numerical(df: pd.DataFrame) -> Dict[str, bool]:
    return {col: is_numerical(df[[col]]) for col in df.columns}


def is_allnan(data: Union[pd.DataFrame, pd.Series, np.ndarray]):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.isna().all().all()
    elif isinstance(data, np.ndarray):
        return np.isnan(data).all()
    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or numpy array.")


def update_json(json_path, items: dict):
    """
    """
    if not json_path.parent.is_dir():
        json_path.parent.mkdir(parents=True, exist_ok=True)
    if not json_path.is_file():
        with open(json_path, 'w+') as file:
            json.dump({}, file)

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    json_data.update(items)
    try:
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)
    except KeyboardInterrupt:
        info_io("Finishing JSON operation before interupt.")
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)

    return json_data


def load_json(json_path):
    """
    """
    if not json_path.is_file():
        return {}

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    return json_data


def write_json(json_path, json_data):
    """
    """
    try:
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)
    except KeyboardInterrupt:
        info_io("Finishing JSON operation before interupt.")
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)

    return


def dict_subset(dictionary: dict, keys: list):
    """_summary_

    Args:
        dictionary (dict): _description_
        keys (list): _description_
    """
    return {k: dictionary[k] for k in keys if k in dictionary}


def is_iterable(obj):
    """_summary_

    Args:
        obj (_type_): _description_

    Returns:
        _type_: _description_
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
