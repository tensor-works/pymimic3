import time
import pickle
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from utils.IO import *
from utils import is_allnan
from preprocessing import AbstractScikitProcessor as AbstractImputer


class PartialImputer(SimpleImputer, AbstractImputer):

    def __init__(self,
                 missing_values=np.nan,
                 strategy='mean',
                 verbose=1,
                 copy=True,
                 storage_path=None):
        """_summary_

        Args:
            storage_path (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 1.
        """
        self._storage_path = storage_path
        self._verbose = verbose
        self.statistics_ = None
        self.n_features_in_ = 0
        self.n_samples_in_ = 0
        super().__init__(missing_values=missing_values, strategy=strategy, copy=copy)

    def partial_fit(self, X):
        """_summary_

        Args:
            X (_type_): _description_
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
        return []
