"""Preprocessing file

This file provides the implemented preprocessing functionalities.

Todo:
    - Use a settings.json
    - implement optional history obj to keept track of the preprocessing history
    - does the interpolate function need to be able to correct time series with no value?
    - Fix categorical data abuse
"""

import numpy as np
import pandas as pd
import random

from copy import deepcopy
from pathlib import Path
from utils.IO import *
from preprocessing.scalers import AbstractScaler
from datasets.trackers import PreprocessingTracker
from datasets.readers import ProcessedSetReader
from tests.settings import *


class AbstractGenerator(object):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 shuffle: bool = True):
        super().__init__()
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._reader = reader
        self._tracker = PreprocessingTracker(storage_path=Path(reader.root_path, "progress"))
        self._steps = self._count_batches()
        self._subject_ids = reader.subject_ids
        self._scaler = scaler
        self._remaining_ids = deepcopy(self._reader.subject_ids)
        if not self._scaler.load():
            self._scaler.fit_reader(reader)
        self.generator = self._generator()

    @property
    def steps(self):
        return self._steps

    def __getitem__(self, index=None):
        X_batch, y_batch = list(), list()
        for _ in range(self._batch_size):
            X, y = next(self.generator)
            X_batch.append(X)
            y_batch.append(y)
        X_batch = self._zeropad_samples(X_batch)
        y_batch = np.array(y_batch)
        return X_batch.astype(np.float32), y_batch.astype(np.float32)

    def _count_batches(self):
        """
        """
        return int(
            np.floor(
                sum([
                    self._tracker.subjects[subject_id]["total"]
                    for subject_id in self._reader.subject_ids
                ])) / self._batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self._steps

    def on_epoch_end(self):
        ...

    def _generator(self):

        while True:
            X_subject, y_subject = self._reader.random_samples().values()
            for X_stay, y_stay in zip(X_subject, y_subject):
                X_stay = self._scaler.transform(X_stay)

                Xs, ys, ts = self.read_timeseries(X_stay, y_stay)

                (Xs, ys, ts) = self._shuffled_data([Xs, ys, ts])

                index = 0
                while index < len(ys):
                    yield Xs[index], ys[index]
                    index += 1

    @staticmethod
    def read_timeseries(X_frame, y_df: pd.DataFrame):
        """
        """
        Xs = list()
        ys = list()
        ts = list()

        for index in range(len(y_df)):

            if index < 0 or index >= len(y_df):
                raise ValueError(
                    "Index must be from 0 (inclusive) to number of examples (exclusive).")

            t = y_df.index[-index - 1]
            y = y_df.iloc[-index - 1, 0]
            if index:
                X = X_frame[:-index, :]
            else:
                X = X_frame

            Xs.append(X)
            ys.append(y)
            ts.append(t)

        return Xs, ys, ts

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

        if self._shuffle:
            random.shuffle(data)

        residual_length = len(data) % self._batch_size
        head = data[:len(data) - residual_length]
        residual = data[len(data) - residual_length:]

        head.sort(key=(lambda x: x[0].shape[0]))

        batches = [head[i:i + self._batch_size] for i in range(0, len(head), self._batch_size)]

        if self._shuffle:
            random.shuffle(batches)

        data = list()

        for batch in batches:
            data += batch

        data += residual
        data = list(zip(*data))

        return data

    @staticmethod
    def _zeropad_samples(data):
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
