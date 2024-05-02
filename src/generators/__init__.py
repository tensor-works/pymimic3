"""Preprocessing file

This file provides the implemented preprocessing functionalities.

Todo:
    - Use a settings.json
    - implement optional history obj to keept track of the preprocessing history
    - does the interpolate function need to be able to correct time series with no value?
    - Fix categorical data abuse
"""

from os import cpu_count
import random
import pandas as pd
import numpy as np
import pdb
import threading
from utils.IO import *


class AbstractBatchGenerator():
    """
    """

    def __init__(self,
                 discretizer: object,
                 type: str,
                 normalizer: object = None,
                 batch_size: int = 8,
                 subset_size: int = None,
                 shuffle: bool = True):
        """_summary_

        Args:
            X (list): _description_
            y (list): _description_
            discretizer (object): _description_
            type (str): _description_
            normalizer (object, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 8.
            subset_size (int, optional): _description_. Defaults to None.
            shuffle (bool, optional): _description_. Defaults to True.
        """
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()
        self.type = type
        self.subset_size = subset_size
        self.shuffle = shuffle
        self.reset_flag = False

        outtype_switch = {"cat": int, "cont": float}
        self.output_type = outtype_switch[type]

    def discretize_frames(self, X_ts):
        """_summary_

        Args:
            index (int): _description_

        Returns:
            _type_: _description_
        """
        X_ts = self.discretizer.fit_transform(X_ts)

        return X_ts

    def normalize_frames(self, X_ts):
        self.normalizer.feature_names_in_ = X_ts.columns
        return pd.DataFrame(self.normalizer.transform(X_ts), columns=X_ts.columns, index=X_ts.index)

    def reset(self):
        """_summary_
        """
        self.reset_flag = True

    def _generator(self):
        """
        """
        raise NotImplementedError

    def _process_frames(self, Xs: list, ts: list):
        """_summary_

        Args:
            Xs (list): _description_
            ts (list): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError
        return data

    def _shuffled_data(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert len(data) >= 2

        if type(data[0][0]) == pd.DataFrame:
            data[0] = [x.values for x in data[0]]

        data = list(zip(*data))

        if self.shuffle:
            random.shuffle(data)

        residual_length = len(data) % self.batch_size
        head = data[:len(data) - residual_length]
        residual = data[len(data) - residual_length:]

        head.sort(key=(lambda x: x[0].shape[0]))

        batches = [head[i:i + self.batch_size] for i in range(0, len(head), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        data = list()

        for batch in batches:
            data += batch

        data += residual
        data = list(zip(*data))

        return data

    def _zeropad_samples(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        dtype = data[0].dtype
        max_len = max([x.shape[0] for x in data])
        ret = [
            np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)],
                           axis=0) for x in data
        ]
        return np.array(ret)

    def __iter__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.generator

    def next(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with self.lock:
            return next(self.generator)

    def __next__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.next()


class AbstractGenerator(AbstractBatchGenerator):
    """
    """

    def __init__(self,
                 discretizer: object,
                 type: str,
                 normalizer: object = None,
                 batch_size: int = 8,
                 subset_size: int = None,
                 shuffle: bool = True):
        """_summary_

        Args:
            X (list): _description_
            y (list): _description_
            discretizer (object): _description_
            type (str): _description_
            normalizer (object, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 8.
            subset_size (int, optional): _description_. Defaults to None.
            shuffle (bool, optional): _description_. Defaults to True.
        """
        super().__init__(discretizer, type, normalizer, batch_size, subset_size, shuffle)

    def _process_frames(self, Xs: list, ts: list):
        """_summary_

        Args:
            Xs (list): _description_
            ts (list): _description_

        Returns:
            _type_: _description_
        """
        data = [*map(self.discretizer.transform, Xs, ts)]

        return data

    @staticmethod
    def read_timeseries(X_df: pd.DataFrame, y_df: pd.DataFrame):
        """
        """
        Xs = list()
        ys = list()
        ts = list()

        def read_df(y_df, index):
            t = y_df.iloc[index][0]
            y = y_df.iloc[index][1]
            X = X_df[X_df.index < t + 1e-6]
            return t, y, X

        def read_np(y_df, index):
            t = y_df[index, 0]
            y = y_df[index, 1]
            X = X_df[X_df.index < t + 1e-6]
            return t, y, X

        reader_switch = {np.ndarray: read_np, pd.DataFrame: read_df}

        for index in range(len(y_df)):

            if index < 0 or index >= len(y_df):
                raise ValueError(
                    "Index must be from 0 (inclusive) to number of examples (exclusive).")

            t, y, X = reader_switch[type(y_df)](y_df, index)

            Xs.append(X)
            ys.append(y)
            ts.append(t)

        return Xs, ys, ts
