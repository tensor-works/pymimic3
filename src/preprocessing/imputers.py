import time
import pickle
import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.impute import SimpleImputer
from utils.IO import *
from utils.arrays import is_allnan
from preprocessing import AbstractScikitProcessor as AbstractImputer


class PartialImputer(SimpleImputer, AbstractImputer):

    def __init__(self,
                 missing_values=np.nan,
                 strategy='mean',
                 verbose=1,
                 copy=True,
                 fill_value=0,
                 add_indicator=False,
                 keep_empty_features=True,
                 storage_path=None):
        """
        An imputer for handling missing values in a dataset with the capability of partial fitting.

        This class extends SimpleImputer from sklearn and AbstractScikitProcessor to handle missing
        values using various strategies and allows for incremental fitting.

        Parameters
        ----------
        missing_values : any, optional
            The placeholder for the missing values. All occurrences of `missing_values` will be
            imputed. Default is np.nan.
        strategy : str, optional
            The imputation strategy. Default is 'mean'.
        verbose : int, optional
            The verbosity level. Default is 1.
        copy : bool, optional
            If True, a copy of X will be created. If False, imputation will be done in-place whenever possible.
            Default is True.
        fill_value : any, optional
            When strategy="constant", `fill_value` is used to replace all occurrences of missing_values.
            Default is 0.
        add_indicator : bool, optional
            If True, a MissingIndicator transform will stack onto the output of the imputer’s transform.
            Default is False.
        keep_empty_features : bool, optional
            If True, features that are all-NaN will be kept in the resulting dataset.
            Default is True.
        storage_path : Path or str, optional
            The path where the imputer's state will be stored. Default is None.
        """
        self._verbose = verbose
        self.statistics_ = None
        self.n_features_in_ = 0
        self.n_samples_in_ = 0
        self._storage_name = "partial_imputer.pkl"
        if storage_path is not None:
            self._storage_path = Path(storage_path, self._storage_name)
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._storage_path = None
        super().__init__(missing_values=missing_values,
                         strategy=strategy,
                         copy=copy,
                         fill_value=fill_value,
                         add_indicator=add_indicator,
                         keep_empty_features=keep_empty_features)

    def partial_fit(self, X):
        """
        Incrementally fit the imputer on a batch of data.

        This method allows the imputer to be fitted in increments, which is useful for large datasets
        that do not fit into memory.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to fit.

        Returns
        -------
        self : object
            Returns self.
        """
        warnings.filterwarnings("ignore")
        n = len(X)
        if is_allnan(X):
            return self
        avg = np.nanmean(X, axis=0)

        if self.statistics_ is None:
            self.fit(X)
            self.n_samples_in_ = n
        else:
            self.statistics_ = np.nanmean([self.n_samples_in_ * self.statistics_, n * avg],
                                          axis=0) / (self.n_samples_in_ + n)
        self.n_samples_in_ += n
        warnings.filterwarnings("default")
        return self

    @classmethod
    def _get_param_names(cls):
        """Necessary for parent class.
        """
        return []
