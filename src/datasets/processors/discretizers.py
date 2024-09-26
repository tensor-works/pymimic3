"""
This module provides the MIMICDiscretizer class to discretize time series data from the MIMIC-III dataset, 
applying various imputation strategies, binning the data into specified time steps, and one-hot encoding 
categorical data.

Usage Examples
--------------
.. code-block:: python

    from pathlib import Path
    from datasets.readers import ProcessedSetReader
    from datasets.trackers import PreprocessingTracker
    from datasets.processors import MIMICDiscretizer

    # Define the path to the dataset, storage, and configuration file
    dataset_path = Path("/path/to/processed/dataset")
    storage_path = Path("/path/to/store/discretized/data")

    # Initialize the reader and tracker
    reader = ProcessedSetReader(dataset_path)
    tracker = PreprocessingTracker(storage_path)

    # Initialize the MIMICDiscretizer for the IHM (In-Hospital Mortality) task
    # Tasks are IHM, DECOMP, LOS, PHENO
    discretizer = MIMICDiscretizer(
        task="IHM",
        reader=reader,
        storage_path=storage_path,
        tracker=tracker,
        time_step_size=1.0,
        start_at_zero=True,
        impute_strategy="previous",
        mode="legacy",
        verbose=True
    )

    # Transform a subject
    subject_id = 12345
    X, y = discretizer.transform_subject(subject_id)

    # Transform the entire dataset
    dataset = reader.read_samples(read_ids=True)
    X = discretizer.transform(dataset, None)

    # Transform the reader directly
    reader = discretizer.transform_reader(reader)

    # Save the transformed data
    discretizer.save_data()
"""

import pandas as pd
import numpy as np
import os
import json
import datetime
from multiprocess import Manager
from pathlib import Path
from typing import Dict, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from settings import *
from utils.jsons import dict_subset
from utils.IO import *
from datasets.readers import ProcessedSetReader
from datasets.writers import DataSetWriter
from datasets.mimic_utils import convert_dtype_value
from datasets.trackers import PreprocessingTracker
from datasets.processors import AbstractProcessor


class MIMICDiscretizer(AbstractProcessor):
    """
    Discretize batch data provided by the reader.

    This class is responsible for discretizing time series data for a specific task,
    provided by a `ProcessedSetReader`, and saving the processed data to a specified
    storage path. The discretizer can handle various imputation strategies and modes
    for processing, bins the data into specified time step intervals and on-hot encodes 
    categorical data.

    Parameters
    ----------
    task : str
        The name of the task. Must be one of the predefined `TASK_NAMES`.
    reader : ProcessedSetReader
        The reader object used to read the batch data.
    tracker : PreprocessingTracker
        The tracker object used to track preprocessing steps.
    storage_path : Path
        The path where the discretized data will be stored.
    time_step_size : float, optional
        The size of each time step for discretization, by default None.
    start_at_zero : bool, optional
        Whether to start the time index at zero, by default True.
    impute_strategy : str, optional
        The strategy for imputing missing values. Can be one of 'normal', 'previous', 'next', or 'zero'.
        Default is 'previous'.
    mode : str, optional
        The mode of discretization. Can be either 'legacy' or 'experimental'. Default is 'legacy'.
    eps : float, optional
        A small value to avoid division by zero errors, by default 1e-6.
    verbose : bool, optional
        If True, print verbose logs during processing. Default is False.
    """

    def __init__(self,
                 task: str,
                 reader: ProcessedSetReader = None,
                 storage_path: Path = None,
                 tracker: PreprocessingTracker = None,
                 time_step_size: float = 1.0,
                 start_at_zero: bool = True,
                 impute_strategy: str = "previous",
                 deep_supervision: bool = False,
                 mode: str = "legacy",
                 eps: float = 1e-6,
                 verbose: bool = False):

        self._operation_name = "discretizing"  # For printing
        self._operation_adjective = "discretized"
        self._save_file_type = "hdf5"
        self._storage_path = storage_path
        if deep_supervision and not task in ["DECOMP", "LOS"]:
            raise ValueError(f"Deep supervision not implemented for {task}")

        self._deep_supervision = deep_supervision
        self._writer = (None if storage_path is None else DataSetWriter(self._storage_path))
        self._source_reader = reader
        if tracker is not None:
            self._tracker = tracker
        else:
            if storage_path is None:
                self._tracker = None
            else:
                self._tracker = PreprocessingTracker(Path(storage_path, "progress"),
                                                     time_step_size=time_step_size,
                                                     start_at_zero=start_at_zero,
                                                     impute_strategy=impute_strategy,
                                                     mode=mode,
                                                     deep_supervision=deep_supervision)
                if len(self._tracker.supervision_modes):
                    self._process_x = False
                else:
                    self._process_x = True
        self._lock = Manager().Lock()
        self._verbose = verbose
        self._discretized_reader = (None if storage_path is None else ProcessedSetReader(
            root_path=storage_path))

        self._time_step_size = time_step_size
        self._start_at_zero = start_at_zero
        self._eps = eps
        if impute_strategy not in ["normal", "previous", "next", "zero"]:
            raise ValueError(
                f"Impute strategy must be one of 'normal', 'previous', 'zero' or 'next'. Impute strategy is {impute_strategy}"
            )
        self._impute_strategy = impute_strategy
        if not mode in ["legacy", "experimental"]:
            raise ValueError(f"Mode must be one of 'legacy' or 'experimental'. Mode is {mode}")
        if mode == "experimental":
            raise NotADirectoryError("Implemented but untested. Will yield nan values.")
        self._mode = mode
        if not task in TASK_NAMES:
            raise ValueError(f"Task name must be one of {TASK_NAMES}. Task name is {task}")
        self._task = task

        with open(Path(os.getenv("CONFIG"), "datasets.json")) as file:
            config_dictionary = json.load(file)
            self._dtypes = config_dictionary["timeseries"]["dtype"]
        with open(Path(os.getenv("CONFIG"), "discretizer_config.json")) as file:
            config_dictionary = json.load(file)
            self._possible_values = config_dictionary['possible_values']
            self._is_categorical = config_dictionary['is_categorical_channel']
            self._impute_values = config_dictionary['normal_values']

        # Tracking variables
        self._init_tracking_variables()

        self._X = dict()
        self._y = dict()
        self._M = dict()

        self._dtype = np.int64 if self._task == 'DECOMP' else np.float32

    @property
    def tracker(self) -> PreprocessingTracker:
        """
        Get the preprocessing tracker object.

        This tracker keeps a record of the preprocessing steps applied to the data.

        Returns
        -------
        PreprocessingTracker
            The preprocessing tracker.
        """
        return self._tracker

    @property
    def subjects(self) -> list:
        """
        Get the list of subject IDs available in the reader.

        This property retrieves the subject IDs from the reader object.

        Returns
        -------
        list
            A list of subject IDs.
        """
        if self._source_reader is None:
            return []
        return self._source_reader.subject_ids

    def _transform(self,
                   dataset: Tuple[Dict[int, Dict[int, pd.DataFrame]]],
                   return_tracking: bool = False):
        """
        Transform the entire dataset when passed as dictionary pair.

        This method processes the entire dataset by discretizing the time series data
        and applying the specified imputation strategy.

        Parameters
        ----------
        X_dict : dict
            A dictionary containing the input data, with subject IDs as keys.
        y_dict : dict
            A dictionary containing the output data, with subject IDs as keys.

        Returns
        -------
        dict
            A dictionary containing the discretized data, with subject IDs as keys.
        """
        X_dict, y_dict = dataset["X"], dataset["y"]
        tracking_info = dict()
        info_io(
            f"Discretizing processed data:\n"
            f"Discretized subjects: {self._n_subjects}\n"
            f"Discretized stays: {self._n_stays}\n"
            f"Discretized samples: {self._n_samples}\n"
            f"Skipped subjects: {self._n_skip}",
            verbose=self._verbose)

        self._samples_processed = 0
        set_process_x_to_false = False

        for subject_id in X_dict.keys():
            X_subject = X_dict[subject_id]
            self._X[subject_id] = dict()
            self._y[subject_id] = dict()
            tracking_info[subject_id] = dict()
            if self._deep_supervision:
                self._M[subject_id] = dict()
            if not self._process_x:
                with self._lock:
                    if subject_id in self._tracker.subject_ids:
                        # Reason for missing subject can be config
                        self._X[subject_id], _ = self._discretized_reader.read_sample(subject_id)
                    else:
                        set_process_x_to_false = True
                        self._process_x = True

            for stay_id in X_subject:
                # Do not reprocess if create for different supervision mode
                y_df = y_dict[subject_id][stay_id]
                if self._process_x:
                    X_df = X_subject[stay_id]
                    if self._mode == "experimental" and self._impute_strategy \
                       in ["previous", "next"]:
                        X_df = self._impute_data(X_df)
                        X_df = self._categorize_data(X_df)
                        X_df = self._bin_data(X_df, y_df)
                    else:
                        X_df = self._categorize_data(X_df)
                        X_df = self._bin_data(X_df, y_df)
                        X_df = self._impute_data(X_df)
                    # Torch friendly dtype
                    self._X[subject_id][stay_id] = X_df.astype(np.float32)
                if self._task == "MULTI":
                    # TODO! other conversions might be necessary
                    # TODO! implement ds for this
                    self._y[subject_id][stay_id] = y_df
                else:
                    if self._deep_supervision:
                        y_reindexed = y_df.reindex(self._X[subject_id][stay_id].index)
                        self._y[subject_id][stay_id] = y_reindexed.fillna(0).astype(np.float32)
                        self._M[subject_id][stay_id] = (~y_reindexed.isna()).astype(bool)
                    else:
                        self._y[subject_id][stay_id] = y_df.astype(np.float32)

                # Based on y_dict not self._y so supervision mode agnostic
                n_samples = len(y_df)
                if n_samples and len(X_df):
                    self._n_stays += 1
                    self._n_samples += n_samples
                    tracking_info[subject_id][stay_id] = n_samples
                else:
                    del self._y[subject_id][stay_id]
                    del self._X[subject_id][stay_id]
                    if self._deep_supervision:
                        del self._M[subject_id][stay_id]

                info_io(
                    f"Discretizing processed data:\n"
                    f"Discretized subjects: {self._n_subjects}\n"
                    f"Discretized stays: {self._n_stays}\n"
                    f"Discretized samples: {self._n_samples}\n"
                    f"Skipped subjects: {self._n_skip}",
                    flush_block=True,
                    verbose=self._verbose)

            tracking_info = self._update_tracking(subject_id, tracking_info, overwrite=False)

            if set_process_x_to_false:
                self._process_x = False

        info_io(
            f"Discretizing processed data:\n"
            f"Discretized subjects: {self._n_subjects}\n"
            f"Discretized stays: {self._n_stays}\n"
            f"Discretized samples: {self._n_samples}\n"
            f"Skipped subjects: {self._n_skip}",
            flush_block=True,
            verbose=self._verbose)

        result_list = [dict_subset(self._X, [subject_id]), \
                       dict_subset(self._y, [subject_id])]

        if self._deep_supervision:
            result_list.append(dict_subset(self._M, [subject_id]))
        if return_tracking:
            return tuple(result_list), tracking_info
        return tuple(result_list)

    def _bin_data(self, X, y):
        """
        Bin the time series data into discrete time steps.
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
                N_bins = int(tsid_to_bins[-1]) + 1
            else:
                N_bins = int(tsid_to_bins[-1] - tsid_to_bins[0]) + 1

            # Reduce DataFrame to bins, keep original channels
            X['bins'] = tsid_to_bins
            X = X.groupby('bins').last()
            X = X.reindex(range(N_bins))

        # return categorized_data
        return X

    def _impute_data(self, X):
        """
        Impute missing values in the time series data using the specified
        imputation strategy.
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

    def _categorize_data(self, X):
        """
        This method transforms categorical variables in the data into a format suitable
        for machine learning models by applying one-hot encoding.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with categorical variables.

        Returns
        -------
        pd.DataFrame
            The data with categorical variables transformed into one-hot encoded format.
        """
        categorized_data = X

        for column in X:
            if not self._is_categorical[column]:
                continue

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
