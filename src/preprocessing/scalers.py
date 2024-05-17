"""Preprocessing file

This file provides the implemented preprocessing functionalities.

Todo:
    - Use a settings.json
    - implement optional history obj to keept track of the preprocessing history
    - does the interpolate function need to be able to correct time series with no value?
    - Fix categorical data abuse
"""
import pdb
import os
import time
import numpy as np
import pickle
from typing import Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from pathos.multiprocessing import cpu_count
from utils.IO import *
from tensorflow.keras.utils import Progbar
from datasets.readers import ProcessedSetReader
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod

__all__ = [
    "AbstractScaler", "MIMICStandardScaler", "MIMICMinMaxScaler", "MIMICMaxAbsScaler",
    "MIMICRobustScaler"
]


class AbstractScaler(ABC):

    @abstractmethod
    def __init__(self, storage_path: Path):
        """_summary_

        Args:
            storage_path (_type_): _description_
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray):
        ...

    @abstractmethod
    def fit(self, X: np.ndarray):
        ...

    @abstractmethod
    def partial_fit(self, X: np.ndarray):
        ...

    def save(self, storage_path=None):
        """_summary_
        """
        if storage_path is not None:
            self._storage_path = Path(storage_path, "scaler.pkl")
        if self._storage_path is None:
            raise ValueError("No storage path provided!")
        with open(self._storage_path, "wb") as save_file:
            pickle.dump(obj=self.__dict__, file=save_file, protocol=2)

    def load(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self._storage_path is not None:
            if self._storage_path.is_file():
                if os.path.getsize(self._storage_path) > 0:
                    with open(self._storage_path, "rb") as load_file:
                        load_params = pickle.load(load_file)
                    for key, value in load_params.items():
                        setattr(self, key, value)

                    return 1
        return 0

    def fit_dataset(self, X):
        """_summary_

        Args:
            discretizer (_type_): _description_
            X (_type_): _description_
        """
        if self._verbose:
            info_io(f"Fitting scaler to dataset of size {len(X)}")
            progbar = Progbar(len(X), unit_name='step')
        n_fitted = 0

        for frame in X:
            self.partial_fit(frame)
            n_fitted += 1
            if self._verbose:
                progbar.update(n_fitted)

        if self._storage_path:
            self.save()

        if self._verbose:
            info_io(f"Done computing new normalizer.")
        return self

    def fit_reader(self, reader: ProcessedSetReader):
        """_summary_

        Args:
            discretizer (_type_): _description_
            reader (_type_): _description_
        """
        if self._verbose:
            info_io(f"Fitting scaler to reader of size {len(reader.subject_ids)}")
            progbar = Progbar(len(reader.subject_ids), unit_name='step')

        n_fitted = 0

        for subject_id in reader.subject_ids:
            X_subjects, _ = reader.read_sample(subject_id).values()
            for frame in X_subjects:
                self.partial_fit(frame)
            n_fitted += 1
            if self._verbose:
                progbar.update(n_fitted)
        if self._storage_path is None:
            self.save(reader.root_path)
        else:
            self.save()

        if self._verbose:
            info_io(f"Done computing new normalizer.\nSaved in location {self._storage_path}!")

        return self


class MIMICStandardScaler(StandardScaler, AbstractScaler):
    """
    """

    def __init__(self, storage_path=None, copy=True, with_mean=True, with_std=True, verbose=True):
        """_summary_

        Args:
            storage_path (_type_): _description_
            copy (bool, optional): _description_. Defaults to True.
            with_mean (bool, optional): _description_. Defaults to True.
            with_std (bool, optional): _description_. Defaults to True.
        """
        self._verbose = verbose
        if storage_path is not None:
            self._storage_path = Path(storage_path, "scaler.pkl")
        else:
            self._storage_path = None
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    @classmethod
    def _get_param_names(cls):
        return []


class MIMICMinMaxScaler(MinMaxScaler, AbstractScaler):
    """
    """

    def __init__(self, storage_path=None, verbose=True):
        """_summary_

        Args:
            storage_path (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 1.
        """
        if storage_path is not None:
            self._storage_path = Path(storage_path, "scaler.pkl")
        else:
            self._storage_path = None
        self._verbose = verbose
        super().__init__()

    @classmethod
    def _get_param_names(cls):
        return []


class MIMICMaxAbsScaler(MaxAbsScaler, AbstractScaler):

    def __init__(self, storage_path=None, verbose=True):
        self._verbose = verbose
        if storage_path is not None:
            self._storage_path = Path(storage_path, "scaler.pkl")
        else:
            self._storage_path = None
        super().__init__()

    @classmethod
    def _get_param_names(cls):
        return []


class MIMICRobustScaler(RobustScaler, AbstractScaler):

    def __init__(self,
                 storage_path=None,
                 with_centering=True,
                 with_scaling=True,
                 quantile_range=(25.0, 75.0),
                 copy=True,
                 verbose=True):
        self._verbose = verbose
        if storage_path is not None:
            self._storage_path = Path(storage_path, "scaler.pkl")
        else:
            self._storage_path = None
        super().__init__(with_centering=with_centering,
                         with_scaling=with_scaling,
                         quantile_range=quantile_range,
                         copy=copy)

    @classmethod
    def _get_param_names(cls):
        return []
