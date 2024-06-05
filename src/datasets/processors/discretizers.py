import pandas as pd
import numpy as np
import os
import json
import datetime
from multiprocess import Manager
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from utils import dict_subset

from settings import *
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
                 time_step_size: float = None,
                 start_at_zero: bool = True,
                 impute_strategy: str = "previous",
                 mode: str = "legacy",
                 eps: float = 1e-6,
                 verbose: bool = False):
        self._storage_path = storage_path
        self._writer = (None if storage_path is None else DataSetWriter(self._storage_path))
        self._source_reader = reader
        if tracker is not None:
            self._tracker = tracker
        else:
            self._tracker = (None if storage_path is None else PreprocessingTracker(
                Path(storage_path, "progress")))
        self._lock = Manager().Lock()
        self._verbose = verbose
        self._discretized_reader = (None if storage_path is None else ProcessedSetReader(
            root_path=storage_path))

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

    def save_data(self, subjects: list = None) -> None:
        """
        Save the discretized data to the storage path.

        If no subjects are specified, all the discretized data will be saved.

        Parameters
        ----------
        subjects : list, optional
            A list of subject IDs to save data for. If None, all data is saved. Default is None.
        """
        if self._writer is None:
            info_io("No storage path provided. Data will not be saved.")
            return
        with self._lock:
            if subjects is None:
                self._writer.write_bysubject({"X": self._X_discretized}, file_type="hdf5")
                self._writer.write_bysubject({"y": self._y_discretized}, file_type="hdf5")
            else:
                self._writer.write_bysubject({"X": dict_subset(self._X_discretized, subjects)},
                                             file_type="hdf5")
                self._writer.write_bysubject({"y": dict_subset(self._y_discretized, subjects)},
                                             file_type="hdf5")

        return

    def transform_subject(self, subject_id: int):
        """
        Transform the data for a specific subject.

        This method reads the data for a specific subject, processes it, and returns
        the discretized data along with tracking information.

        Parameters
        ----------
        subject_id : int
            The ID of the subject to transform data for.

        Returns
        -------
        tuple
            A tuple containing the discretized data and the tracking information.
        """
        X_processed, y_processed = self._source_reader.read_sample(subject_id,
                                                                   read_ids=True,
                                                                   data_type=pd.DataFrame).values()
        X = {subject_id: X_processed}
        y = {subject_id: y_processed}

        X_discretized = self.transform(X, y)
        if X_discretized is None:
            return None, None
        if self._tracker is None:
            return X_discretized, y

        with self._lock:
            tracking_info = self._tracker.subjects[subject_id]
        return (X_discretized, y), tracking_info

    def transform(self, X_dict, y_dict):
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
        n_subjects = 0
        n_stays = 0
        n_samples = 0
        n_skip = 0

        if self._verbose:
            info_io(f"Discretizing processed data:\n"
                    f"Discretized subjects: {0}\n"
                    f"Discretized stays: {0}\n"
                    f"Discretized samples: {0}\n"
                    f"Skipped subjects: {0}")

        self._samples_processed = 0

        self._X_discretized = dict()
        self._y_discretized = dict()

        for subject_id in X_dict.keys():
            X_subject = X_dict[subject_id]
            self._X_discretized[subject_id] = dict()
            self._y_discretized[subject_id] = dict()
            tracking_info = dict()

            for stay_id in X_subject:
                X_df = X_subject[stay_id]
                if self._mode == "experimental" and self._impute_strategy in ["previous", "next"]:
                    X_df = self._impute_data(X_df)
                    X_df = self._categorize_data(X_df)
                    X_df = self._bin_data(X_df)
                else:
                    X_df = self._categorize_data(X_df)
                    X_df = self._bin_data(X_df)
                    X_df = self._impute_data(X_df)
                self._X_discretized[subject_id][stay_id] = X_df
                self._y_discretized[subject_id][stay_id] = y_dict[subject_id][stay_id]

                tracking_info[stay_id] = len(y_dict[subject_id][stay_id])

                if self._verbose:
                    info_io(
                        f"Discretizing processed data:\n"
                        f"Discretized subjects: {n_subjects}\n"
                        f"Discretized stays: {n_stays}\n"
                        f"Discretized samples: {n_samples}"
                        f"Skipped subjects: {n_skip}",
                        flush_block=True)

            n_subjects += 1
            if self._tracker is not None:
                with self._lock:
                    self._tracker.subjects.update({subject_id: tracking_info})

            if not len(self._y_discretized[subject_id]) or not len(self._X_discretized[subject_id]):
                del self._y_discretized[subject_id]
                del self._X_discretized[subject_id]
                n_skip += 1
            else:
                n_subjects += 1

        if self._verbose:
            info_io(
                f"Discretizing processed data:\n"
                f"Discretized subjects: {n_subjects}\n"
                f"Discretized stays: {n_stays}\n"
                f"Discretized samples: {n_samples}"
                f"Skipped subjects: {n_skip}",
                flush_block=True)

        return self._X_discretized

    def _bin_data(self, X):
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
                N_bins = tsid_to_bins[-1] + 1
            else:
                N_bins = tsid_to_bins[-1] - tsid_to_bins[0] + 1

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
