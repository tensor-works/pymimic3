"""Dataset file

This file allows access to the dataset as specified.

Todo:

YerevaNN/mimic3-benchmarks
"""
import numpy as np
import pandas as pd
import json
from scipy.stats import skew
from numpy import random
from multiprocess import Manager
from datasets.readers import ProcessedSetReader
from datasets.writers import DataSetWriter
from utils import dict_subset
from utils.IO import *
from pathlib import Path
from datasets.trackers import PreprocessingTracker
from . import AbstractProcessor


class MIMICFeatureEngine(AbstractProcessor):
    """_summary_
    """

    def __init__(self,
                 config_dict: Path,
                 task: str,
                 reader: ProcessedSetReader = None,
                 storage_path: Path = None,
                 tracker: PreprocessingTracker = None,
                 verbose=False) -> None:
        """_summary_

        Args:
            config_dict (Path): _description_
            task (str): _description_
            storage_path (Path, optional): _description_. Defaults to None.
            source_path (Path, optional): _description_. Defaults to None.
            tracking (bool, optional): _description_. Defaults to True.
            save_as_sample (bool, optional): _description_. Defaults to True.
        """
        self._storage_path = storage_path
        self._writer = (DataSetWriter(storage_path) if storage_path is not None else None)
        self._reader = reader
        if tracker is not None:
            self._tracker = tracker
        else:
            self._tracker = (PreprocessingTracker(Path(storage_path, "progress"))
                             if storage_path is not None else None)
        self._task = task
        self._save_as_samples = (True if task in ["IHM", "PHENO"]\
                                 else False)
        self._subsample_switch = {
            "first_percentage":
                lambda start_t, end_t, percentage: (start_t, start_t +
                                                    (end_t - start_t) * percentage / 100.0),
            "last_percentage":
                lambda start_t, end_t, percentage: (end_t -
                                                    (end_t - start_t) * percentage / 100.0, end_t)
        }

        self._lock = Manager().Lock()
        self._verbose = verbose

        with open(config_dict) as file:
            config_dict = json.load(file)
            self._sampler_combinations = config_dict["sampler_combinations"]
            self._impute_config = config_dict["channels"]
            self._channel_names = config_dict["channel_names"]

    @property
    def subjects(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        return self._reader.subject_ids

    def transform_subject(self, subject_id: int):
        X_processed, y_processed = self._reader.read_sample(subject_id,
                                                            read_ids=True,
                                                            data_type=pd.DataFrame).values()
        X = {subject_id: X_processed}
        y = {subject_id: y_processed}
        if X is None or y is None:
            return None, None

        X_engineered, y_engineered, _ = self.transform(X, y)
        if X_engineered is None or y_engineered is None:
            return None, None
        if self._tracker is None:
            return X_engineered, y_engineered

        with self._lock:
            tracking_info = self._tracker.subjects[subject_id]
        return (X_engineered, y_engineered), tracking_info

    def transform(self, X_dict: dict, y_dict: dict):
        """_summary_

        Args:
            X_dict (dict): _description_
            y_dict (dict): _description_

        Returns:
            _type_: _description_
        """
        n_subjects = 0
        n_stays = 0
        n_samples = 0

        if self._verbose:
            info_io(f"Engineering processed data:\n"
                    f"Engineered subjects: {0}\n"
                    f"Engineered stays: {0}\n"
                    f"Engineered samples: {0}")

        self._samples_processed = 0

        self._X_processed = dict()
        self._y_processed = dict()
        self._t_processed = dict()

        for subject_id in X_dict.keys():
            X_subject = X_dict[subject_id]
            y_subject = y_dict[subject_id]
            self._X_processed[subject_id] = dict()
            self._y_processed[subject_id] = dict()
            self._t_processed[subject_id] = dict()
            tracking_info = dict()

            for stay_id in X_subject:
                X_df = X_subject[stay_id]
                y_df = y_subject[stay_id]

                X_ss, ys, ts = self._engineer_stay(X_df, y_df)
                X_ss, ys, ts = self._convert_feature_dtype(X_ss, ys, ts)
                self._X_processed[subject_id][stay_id] = X_ss
                self._y_processed[subject_id][stay_id] = np.squeeze(ys)
                self._t_processed[subject_id][stay_id] = ts
                tracking_info[stay_id] = len(ys)
                n_samples += len(ys)
                n_stays += 1

                if self._verbose:
                    info_io(
                        f"Engineering processed data:\n"
                        f"Engineered subjects: {n_subjects}\n"
                        f"Engineered stays: {n_stays}\n"
                        f"Engineered samples: {n_samples}",
                        flush_block=True)

            n_subjects += 1
            if self._tracker is not None:
                with self._lock:
                    self._tracker.subjects.update({subject_id: tracking_info})

        if self._verbose:
            info_io(
                f"Engineering processed data:\n"
                f"Engineered subjects: {n_subjects}\n"
                f"Engineered stays: {n_stays}\n"
                f"Engineered samples: {n_samples}",
                flush_block=True)

        return self._X_processed, self._y_processed, self._t_processed

    def _engineer_stay(self, X_df, y_df):
        X_df = self._make_categorical_data(X_df)
        Xs, ys, ts = self._read_timeseries_windows(X_df, y_df)

        (Xs, ys, ts) = self._shuffle([Xs, ys, ts])

        X_ss = list()
        ys = list(ys)
        ts = list(ts)

        for df in Xs:
            subsamples = [[
                self._channel_subsampler(df[column], *combination)
                for combination in self._sampler_combinations
            ]
                          for column in self._channel_names]
            # Iterating by channel name from config allows normalization
            # and ensures comparability to ground truth data from original dir

            engineered_features = [
                self._make_engineered_features(channel.values)
                for subsample in subsamples
                for channel in subsample
            ]

            X_ss.append(np.concatenate(engineered_features))

            self._samples_processed += 1

        return X_ss, ys, ts

    def _convert_feature_dtype(self, X, y, t):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            t (_type_): _description_
        """
        return X, y, t

    def save_data(self, subject_ids: list = None) -> None:
        """_summary_

        Args:
            storage_path (Path, optional): _description_. Defaults to None.
            task (str, optional): _description_. Defaults to None.
        """
        if subject_ids is None:
            name_data_pairs = {
                "X": self._X_processed,
                "y": self._y_processed,
                "t": self._t_processed
            }
        else:
            name_data_pairs = {
                "X": dict_subset(self._X_processed, subject_ids),
                "y": dict_subset(self._y_processed, subject_ids),
                "t": dict_subset(self._t_processed, subject_ids)
            }
        with self._lock:
            self._writer.write_bysubject(name_data_pairs)

        def create_df(data, file_name) -> pd.DataFrame:
            if file_name == "X":
                dfs = pd.DataFrame([([subject_id, stay_id] +
                                     np.squeeze(frame).tolist()) if len(np.squeeze(frame)) > 1 else
                                    ([subject_id, stay_id, float(frame)])
                                    for subject_id, subject_stays in data.items()
                                    for stay_id, frame in subject_stays.items()])

            elif file_name == "y":
                dfs = pd.DataFrame([([subject_id, stay_id] +
                                     frame.tolist()) if isinstance(frame.tolist(), list) else
                                    ([subject_id, stay_id, float(frame)])
                                    for subject_id, subject_stays in data.items()
                                    for stay_id, frame in subject_stays.items()])
            dfs = dfs.rename(columns={0: "subject_id", 1: "stay_id"})
            if not len(dfs):
                return
            return dfs

        def append_data(X: dict, y: dict):

            def append(dfs: pd.DataFrame, file_name: str):
                file = Path(self._storage_path, f"{file_name}.csv")
                if file.is_file():
                    dfs.to_csv(file, mode='a', header=False, index=False)
                else:
                    dfs.to_csv(file, index=False)

            X_df = create_df(X, "X")
            y_df = create_df(y, "y")
            if y_df is None or not len(y_df) or not len(X_df):
                return
            append(X_df, "X")
            append(y_df, "y")

        if self._save_as_samples:
            with self._lock:
                append_data(self._X_processed, self._y_processed)

        return

    def _shuffle(self, data) -> None:
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert len(data) >= 2

        data = list(zip(*data))
        random.shuffle(data)
        data = list(zip(*data))

        return data

    def _make_categorical_data(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        replace_dict = {'nan': np.nan}

        for channel in self._impute_config.keys():
            if 'values' in self._impute_config[channel].keys():
                replace_dict.update(self._impute_config[channel]['values'])

        return X.replace(replace_dict).astype(float)

    def _make_engineered_features(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        functions = [min, max, np.mean, np.std, skew, len]
        import warnings
        warnings.filterwarnings("error")

        if len(data) == 0:
            engineered_data = np.full((len(functions,)), np.nan)
        else:
            # !TODO DEBUGGING
            engineered_data = [
                fn(data) if fn is not skew or
                (len(data) > 1 and not all(i == data[0]
                                           for i in data) or fn is len) else
                0  #TODO! This will fail and be NaN in Windows
                for fn in functions
            ]
            engineered_data = np.array(engineered_data, dtype=np.float32)

        return engineered_data

    def _channel_subsampler(self, Sr: pd.Series, sampler_function, percentage):
        """_summary_

        Args:
            Sr (pd.Series): _description_
            sampler_function (_type_): _description_
            percentage (_type_): _description_

        Returns:
            _type_: _description_
        """
        Sr = Sr.dropna()

        if len(Sr) == 0:
            return pd.DataFrame()

        start_t = Sr.index[0]
        end_t = Sr.index[-1]

        sampled_start_t, sampled_end_t = self._subsample_switch[sampler_function](start_t, end_t,
                                                                                  percentage)

        return Sr[(Sr.index < sampled_end_t + 1e-6) & (Sr.index > sampled_start_t - 1e-6)]

    def _timeseries_subsampler(self, X: pd.DataFrame, sampler_function, percentage):
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            sampler_function (_type_): _description_
            percentage (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(X) == 0:
            data = np.full((6), np.nan)
        else:
            start_t = X.index[0]
            end_t = X.index[-1]

            sampled_start_t, sampled_end_t = self._subsample_switch[sampler_function](start_t,
                                                                                      end_t,
                                                                                      percentage)

            data = X[(X.index < sampled_end_t + 1e-6) & (X.index > sampled_start_t - 1e-6)]

            if len(data) == 0:
                data = pd.DataFrame(np.full((len(X,)), np.nan))

        return [data[channel] for channel in data]

    def _read_timeseries_windows(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> 'tuple[list]':
        """_summary_

        Args:
            X_df (pd.DataFrame): _description_
            y_df (pd.DataFrame): _description_

        Raises:
            ValueError: _description_

        Returns:
            tuple: _description_
        """
        Xs = list()
        ys = list()
        ts = list()

        for i in range(len(y_df)):
            index = i

            if index < 0 or index >= len(y_df):
                raise ValueError(
                    "Index must be from 0 (inclusive) to number of examples (exclusive).")

            t = y_df.reset_index().iloc[index, 0]
            y = y_df.reset_index().iloc[index, 1:]
            X = X_df[X_df.index < t + 1e-6]

            Xs.append(X)
            ys.append(y)
            ts.append(t)

        return Xs, ys, ts

    def _convert_feature_dtype(self, X, y, t):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            t (_type_): _description_
        """
        X = np.stack(X)
        return np.array(X), np.array(y), np.array(t)
