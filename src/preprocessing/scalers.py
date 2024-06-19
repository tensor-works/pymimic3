import numpy as np
import pandas as pd
from typing import Union
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.preprocessing import RobustScaler as _RobustScaler
from sklearn.preprocessing import MaxAbsScaler as _MaxAbsScaler
from utils.IO import *
from pathlib import Path
from preprocessing import AbstractScikitProcessor

__all__ = ["AbstractScaler", "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler"]


class AbstractScaler(AbstractScikitProcessor):
    """
    Base class for all scalers in the MIMIC-III preprocessing pipeline.

    This class handles initialization of storage paths and optional imputers, along with
    verbosity settings.

    Parameters
    ----------
    storage_path : str or Path, optional
        The path where the scaler's state will be stored, by default None.
    imputer : object, optional
        The imputer to use for handling missing values, by default None. Called
        before fitting or transforming on each batch if passed.
    verbose : bool, optional
        If True, print verbose logs during processing, by default True.
    """

    def __init__(self, storage_path=None, imputer=None, verbose=True):
        if storage_path is not None:
            self._storage_path = Path(storage_path, self._storage_name)
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._storage_path = None
        self._verbose = verbose
        self._imputer = imputer
        self._action = "scaling"


class StandardScaler(_StandardScaler, AbstractScaler):
    """
    """

    def __init__(self,
                 storage_path=None,
                 imputer=None,
                 copy=True,
                 with_mean=True,
                 with_std=True,
                 verbose=True):
        """
        Standard Scaler for the MIMIC-III dataset.

        This scaler standardizes features by removing the mean and scaling to unit variance.

        Parameters
        ----------
        storage_path : str or Path, optional
            The path where the scaler's state will be stored, by default None.
        imputer : object, optional
            The imputer to use for handling missing values, by default None.
        copy : bool, optional
            If True, a copy of X will be created, by default True.
        with_mean : bool, optional
            If True, center the data before scaling, by default True.
        with_std : bool, optional
            If True, scale the data to unit variance, by default True.
        verbose : bool, optional
            If True, print verbose logs during processing, by default True.
        """
        self._name = "standard scaler"
        self._storage_name = "standard_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        _StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)

    @classmethod
    def _get_param_names(cls):
        """
        Necessary for scikit-learn compatibility.
        """
        return []

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Scale the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be scaled.

        Returns
        -------
        np.ndarray
            The scaled data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            X = self._imputer.transform(X)
        return super().transform(X)

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame] = None,
            **fit_params):
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to compute the mean and std on.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.DataFrame] = None,
                      **fit_params) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit_transform(X, y, **fit_params)


class MinMaxScaler(_MinMaxScaler, AbstractScaler):
    """
    Min-Max Scaler for the MIMIC-III dataset.

    This scaler transforms features by scaling each feature to a given range.

    Parameters
    ----------
    storage_path : str or Path, optional
        The path where the scaler's state will be stored, by default None.
    imputer : object, optional
        The imputer to use for handling missing values, by default None.
    verbose : bool, optional
        If True, print verbose logs during processing, by default True.
    feature_range : tuple (min, max), optional
        Desired range of transformed data, by default (0, 1).
    copy : bool, optional
        If True, a copy of X will be created, by default True.
    clip : bool, optional
        Set to True to clip transformed values of held-out data to provided feature range, by default False.
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
        self._name = "min-max scaler"
        self._storage_name = "minmax_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        _MinMaxScaler.__init__(self, feature_range=feature_range, copy=copy, clip=clip)

    @classmethod
    def _get_param_names(cls):
        """
        Necessary for scikit-learn compatibility.
        """
        return []

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Scale the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be scaled.

        Returns
        -------
        np.ndarray
            The scaled data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            X = self._imputer.transform(X)
        return super().transform(X)

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame] = None,
            **fit_params):
        """
        Compute the volumn wise minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to compute the minimum and maximum on.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.DataFrame] = None,
                      **fit_params) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit_transform(X, y, **fit_params)


class MaxAbsScaler(_MaxAbsScaler, AbstractScaler):
    """
    Max-Abs Scaler for the MIMIC-III dataset.

    This scaler scales each feature by its maximum absolute value, preserving the sparsity of the data.

    Parameters
    ----------
    storage_path : str or Path, optional
        The path where the scaler's state will be stored, by default None.
    imputer : object, optional
        The imputer to use for handling missing values, by default None.
    verbose : bool, optional
        If True, print verbose logs during processing, by default True.
    copy : bool, optional
        If True, a copy of X will be created, by default True.
    """

    def __init__(self, storage_path=None, imputer=None, verbose=True, copy=True):
        self._name = "max-abs scaler"
        self._storage_name = "maxabs_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        _MaxAbsScaler().__init__(copy=copy)

    @classmethod
    def _get_param_names(cls):
        """
        Necessary for scikit-learn compatibility.
        """
        return []

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Scale the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be scaled.

        Returns
        -------
        np.ndarray
            The scaled data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            X = self._imputer.transform(X)
        return super().transform(X)

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame] = None,
            **fit_params):
        """
        Compute the maximum absolute value to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to compute the maximum absolute value on.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.DataFrame] = None,
                      **fit_params) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit_transform(X, y, **fit_params)


class RobustScaler(_RobustScaler, AbstractScaler):
    """
    Robust Scaler for the MIMIC-III dataset.

    This scaler scales features using statistics that are robust to outliers.

    Parameters
    ----------
    storage_path : str or Path, optional
        The path where the scaler's state will be stored, by default None.
    imputer : object, optional
        The imputer to use for handling missing values, by default None.
    with_centering : bool, optional
        If True, center the data before scaling, by default True.
    with_scaling : bool, optional
        If True, scale the data to the interquartile range, by default True.
    quantile_range : tuple (float, float), optional
        Quantile range used to calculate the scale, by default (25.0, 75.0).
    copy : bool, optional
        If True, a copy of X will be created, by default True.
    verbose : bool, optional
        If True, print verbose logs during processing, by default True.
    """

    def __init__(self,
                 storage_path=None,
                 imputer=None,
                 with_centering=True,
                 with_scaling=True,
                 quantile_range=(25.0, 75.0),
                 copy=True,
                 verbose=True):
        self._name = "robust scaler"
        self._storage_name = "robust_scaler.pkl"
        AbstractScaler.__init__(self, storage_path=storage_path, imputer=imputer, verbose=verbose)
        _RobustScaler().__init__(with_centering=with_centering,
                                 with_scaling=with_scaling,
                                 quantile_range=quantile_range,
                                 copy=copy)

    @classmethod
    def _get_param_names(cls):
        """
        Necessary for scikit-learn compatibility.
        """
        return []

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Scale the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be scaled.

        Returns
        -------
        np.ndarray
            The scaled data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            X = self._imputer.transform(X)
        return super().transform(X)

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame] = None,
            **fit_params):
        """
        Compute the median and quantiles to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to compute the median and quantiles on.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.DataFrame] = None,
                      **fit_params) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform.
        y : None
            Ignored.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        if hasattr(self, "_imputer") and self._imputer is not None:
            return self._imputer.transform(X)
        return super().fit_transform(X, y, **fit_params)
