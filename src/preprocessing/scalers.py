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
from typing import Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from utils.IO import *
from pathlib import Path
from preprocessing import AbstractScikitProcessor

__all__ = [
    "AbstractScaler", "MIMICStandardScaler", "MIMICMinMaxScaler", "MIMICMaxAbsScaler",
    "MIMICRobustScaler"
]


class AbstractScaler(AbstractScikitProcessor):

    def __init__(self, storage_path=None, imputer=None, verbose=True):
        if storage_path is not None:
            self._storage_path = Path(storage_path, self._storage_name)
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._storage_path = None
        self._verbose = verbose
        self._imputer = imputer

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if hasattr(self, "_imputer") and self._imputer is not None:
            X = self._imputer.transform(X)
        return super().transform(X)

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame] = None,
            **fit_params):
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.DataFrame] = None,
                      **fit_params) -> np.ndarray:
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit_transform(X, y, **fit_params)


class MIMICStandardScaler(StandardScaler, AbstractScaler):
    """
    """

    def __init__(self,
                 storage_path=None,
                 imputer=None,
                 copy=True,
                 with_mean=True,
                 with_std=True,
                 verbose=True):
        """_summary_

        Args:
            storage_path (_type_): _description_
            copy (bool, optional): _description_. Defaults to True.
            with_mean (bool, optional): _description_. Defaults to True.
            with_std (bool, optional): _description_. Defaults to True.
        """
        self._storage_name = "standard_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)

    @classmethod
    def _get_param_names(cls):
        return []


class MIMICMinMaxScaler(MinMaxScaler, AbstractScaler):
    """
    """

    def __init__(self,
                 storage_path=None,
                 imputer=None,
                 verbose=True,
                 feature_range=(0, 1),
                 copy=True,
                 clip=False):
        """_summary_

        Args:
            storage_path (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 1.
        """
        self._storage_name = "minmax_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        MinMaxScaler.__init__(self, feature_range=feature_range, copy=copy, clip=clip)

    @classmethod
    def _get_param_names(cls):
        return []


class MIMICMaxAbsScaler(MaxAbsScaler, AbstractScaler):

    def __init__(self, storage_path=None, imputer=None, verbose=True, copy=True):
        self._storage_name = "maxabs_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        super().__init__(copy=copy)

    @classmethod
    def _get_param_names(cls):
        return []


class MIMICRobustScaler(RobustScaler, AbstractScaler):

    def __init__(self,
                 storage_path=None,
                 imputer=None,
                 with_centering=True,
                 with_scaling=True,
                 quantile_range=(25.0, 75.0),
                 copy=True,
                 verbose=True):
        self._storage_name = "robust_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        super().__init__(with_centering=with_centering,
                         with_scaling=with_scaling,
                         quantile_range=quantile_range,
                         copy=copy)

    @classmethod
    def _get_param_names(cls):
        return []
