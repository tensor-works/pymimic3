import torch
import pandas as pd
import numpy as np
from typing import List
from preprocessing.scalers import AbstractScaler
from datasets.readers import ProcessedSetReader
from torch.utils.data import Dataset
from tests.settings import *
from utils.IO import *
from torch.utils.data import DataLoader
from . import AbstractGenerator


class RiverGenerator(AbstractGenerator, Dataset):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 shuffle: bool = True):
        super(RiverGenerator, self).__init__(reader=reader,
                                             scaler=scaler,
                                             batch_size=1,
                                             shuffle=shuffle)
        self._names: List[str] = None
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self, *args, **kwargs):
        if self._index >= self.steps:
            raise StopIteration
        X, y = super().__getitem__()
        X = np.squeeze(X)
        if self._names is None:
            self._names = [str(i) for i in range(714)]
        X = dict(zip(self._names, X))
        y = float(np.squeeze(y))
        self._index += 1
        return X, y

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

            t = y_df.index[index]
            y = y_df.iloc[index - 1, 0]
            X = X_frame[index, :]

            Xs.append(X)
            ys.append(y)
            ts.append(t)

        return Xs, ys, ts
