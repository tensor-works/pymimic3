"""Preprocessing file

This file provides the implemented preprocessing functionalities.

Todo:
    - Use a settings.json
    - implement optional history obj to keept track of the preprocessing history
    - does the interpolate function need to be able to correct time series with no value?
    - Fix categorical data abuse
"""
import pandas as pd
import numpy as np
import os
import json
import datetime
from pathlib import Path
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from datasets.mimic_utils import convert_dtype_value
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from settings import *
from utils.IO import *


class BatchDiscretizer(object):
    """ Discretize batch data provided by reader.
    """

    def __init__(self,
                 task_name,
                 time_step_size=None,
                 start_at_zero=True,
                 impute_strategy="previous",
                 mode="legacy",
                 eps=1e-6):
        """
        """
        self._time_step_size = time_step_size
        self._start_at_zero = start_at_zero
        self._eps = eps
        if not impute_strategy in ["normal", "previous", "next", "zero"]:
            raise ValueError(
                f"Impute strategy must be one of 'normal', 'previous', 'zero' or 'next'. Impute strategy is {impute_strategy}"
            )
        self._impute_strategy = impute_strategy
        if not mode in ["legacy", "experimental"]:
            raise ValueError(f"Mode must be one of 'legacy' or 'experimental'. Mode is {mode}")
        if mode == "experimental":
            raise NotADirectoryError("Implemented but untested. Will yield nan values.")
        self._mode = mode
        if not task_name in TASK_NAMES:
            raise ValueError(f"Task name must be one of {TASK_NAMES}. Task name is {task_name}")
        self._task_name = task_name

        with open(Path(os.getenv("CONFIG"), "datasets.json")) as file:
            config_dictionary = json.load(file)
            self._dtypes = config_dictionary["timeseries"]["dtype"]
        with open(Path(os.getenv("CONFIG"), "discretizer_config.json")) as file:
            config_dictionary = json.load(file)
            self._possible_values = config_dictionary['possible_values']
            self._is_categorical = config_dictionary['is_categorical_channel']
            self._impute_values = config_dictionary['normal_values']

    def _bin_data(self, X):
        """
        """

        if self._time_step_size is not None:
            # Get data frame parameters
            start_timestamp = (0 if self._start_at_zero else X.index[0])

            if is_datetime(X.index):
                ts = list((X.index - start_timestamp) / datetime.timedelta(hours=1))
            else:
                ts = list(X.index - start_timestamp)

            # Maps sample_periods to discretization bins
            tsid_to_bins = list(map(lambda x: int(x / self._time_step_size - self._eps), ts))
            # Tentative solution
            if self._start_at_zero:
                N_bins = tsid_to_bins[-1] + 1
            else:
                N_bins = tsid_to_bins[-1] - tsid_to_bins[0] + 1

            # Reduce DataFrame to bins, keep original channels
            X['bins'] = tsid_to_bins
            X = X.groupby('bins').last()
            X = X.reindex(range(N_bins))

        # return categorized_data
        return X

    def transform(self, X):
        """
        """
        if self._mode == "experimental" and self._impute_strategy in ["previous", "next"]:
            X = self._impute_data(X)
            X = self._categorize_data(X)
            X = self._bin_data(X)
        else:
            X = self._categorize_data(X)
            X = self._bin_data(X)
            X = self._impute_data(X)

        return X

    def _impute_data(self, X):
        """
        """
        if self._start_at_zero:
            tsid_to_bins = list(map(lambda x: int(x / self._time_step_size - self._eps), X.index))
            start_count = 0
            while not start_count in tsid_to_bins:
                start_count += 1
                X = pd.concat([pd.DataFrame(pd.NA, index=[0], columns=X.columns, dtype="Int32"), X])

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._possible_values.keys())
        if self._impute_strategy == "normal":
            for column in X:
                if column in self._impute_values.keys():
                    X[column] = X[column].replace(
                        np.nan,
                        convert_dtype_value(self._impute_values[column], self._dtypes[column]))
                elif column.split("->")[0] in self._impute_values.keys():
                    cat_name, cat_value = column.split("->")
                    X[column] = X[column].replace(
                        np.nan, (1.0 if cat_value == self._impute_values[cat_name] else 0.0))

        elif self._impute_strategy in ["previous", "next"]:
            X = (X.ffill() if self._impute_strategy == "previous" else X.bfill())
            for column in X:
                if X[column].isnull().values.any():
                    if column in self._impute_values:
                        impute_value = convert_dtype_value(self._impute_values[column],
                                                           self._dtypes[column])
                    else:
                        cat_name, cat_value = column.split("->")
                        impute_value = (1.0 if cat_value == self._impute_values[cat_name] else 0.0)
                    X[column] = X[column].replace(np.nan, impute_value)
        elif self._impute_strategy == "zero":
            X = X.fillna(0)
        return X

    def _impute(self, train_data, imp_strategy='mean'):
        """
        """
        imputer = SimpleImputer(missing_values=np.nan, strategy=imp_strategy)
        imputer.fit(train_data)
        train_data = imputer.transform(train_data)

        return train_data

    def _categorize_data(self, X):
        """
        """
        categorized_data = X

        for column in X:
            if not self._is_categorical[column]:
                continue
            '''
            categories = [str(cat) for cat in self._possible_values[column]]
            # nan_indices = categorized_data[column].isna()
            values = categorized_data[column].dropna().astype('object').values.reshape(-1, 1)
            # values = categorized_data.pop(column).astype(pd.Categorical).values.reshape(-1, 1)
            encoder = OneHotEncoder(categories=list( \
                                    np.array(categories).reshape(1, len(categories))), \
                                    handle_unknown='ignore')
            encoded_channel = encoder.fit_transform(values).toarray()

            nan_indices = base_col.isna()
            '''
            categories = [str(cat) for cat in self._possible_values[column]]

            # Finding nan indices
            nan_indices = categorized_data[column].isna()
            if not nan_indices.all() and nan_indices.any():
                # Handling non-NaN values: encode them
                non_nan_values = categorized_data.pop(column).dropna().astype(
                    'string').values.reshape(-1, 1)
                encoder = OneHotEncoder(categories=list(
                    np.array(categories).reshape(1, len(categories))),
                                        handle_unknown='ignore')
                encoded_values = encoder.fit_transform(non_nan_values).toarray()

                # Initialize an array to hold the final encoded values including NaNs
                encoded_channel = np.full((categorized_data.shape[0], len(categories)), np.nan)

                # Fill in the encoded values at the right indices
                non_nan_indices = ~nan_indices
                encoded_channel[non_nan_indices] = encoded_values
            elif not nan_indices.any():
                encoder = OneHotEncoder(categories=[np.array(categories).reshape(-1)],
                                        handle_unknown='ignore')
                encoded_channel = encoder.fit_transform(
                    categorized_data.pop(column).astype('string').values.reshape(-1, 1)).toarray()
            else:
                categorized_data.pop(column)
                encoded_channel = np.full((categorized_data.shape[0], len(categories)), np.nan)

            categorized_data = pd.concat([\
                categorized_data,
                pd.DataFrame(\
                    encoded_channel,
                    columns=[f"{column}->{category}" for category in categories],
                    index=categorized_data.index)
                ],
                axis=1)

        return categorized_data
