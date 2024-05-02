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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathos.multiprocessing import cpu_count
from pathos.pools import ThreadPool
from utils.IO import *
from tensorflow.keras.utils import Progbar
import pandas as pd


# TODO consider double inheritance pattern here
class Normalizer(StandardScaler):
    """
    """

    def __init__(self, storage_path, copy=True, with_mean=True, with_std=True):
        """_summary_

        Args:
            storage_path (_type_): _description_
            copy (bool, optional): _description_. Defaults to True.
            with_mean (bool, optional): _description_. Defaults to True.
            with_std (bool, optional): _description_. Defaults to True.
        """
        self.storage_path = storage_path
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def save(self):
        """_summary_
        """
        with open(self.storage_path, "wb") as save_file:
            pickle.dump(obj=self.__dict__, file=save_file, protocol=2)

    def load(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.storage_path.is_file():
            if os.path.getsize(self.storage_path) > 0:
                with open(self.storage_path, "rb") as load_file:
                    load_params = pickle.load(load_file)
            else:
                return 0
        else:
            return 0

        for key, value in load_params.items():
            setattr(self, key, value)

        return 1

    def fit_reader(self, reader, discretizer=None, imputer=None):
        """_summary_

        Args:
            discretizer (_type_): _description_
            reader (_type_): _description_
        """
        directories = [path for path in reader.root_path.iterdir() if path.name.isnumeric()]
        directories = directories[:int(len(directories))]

        progbar = Progbar(len(directories), unit_name='step')

        if imputer is None:

            def fit_subject(folder):
                data = reader.read_sample(folder)
                X_subjects, y_subject = data["X"], data["y"]

                if X_subjects is None or y_subject is None:
                    return None

                return [
                    discretizer.make_categorical_data(discretizer.make_imputed_data(frame.reshape(1, -1))) \
                    for frame in X_subjects
                ]
        elif discretizer is None:

            def fit_subject(folder):
                data = reader.read_sample(folder)
                X_subjects, y_subject = data["X"], data["y"]

                if X_subjects is None or y_subject is None:
                    return None

                return [
                    imputer.transform(sample.reshape(1, -1))
                    for frame in X_subjects
                    for sample in frame
                ]
        else:
            raise ValueError("Need imputer or discretizer!")

        pool = ThreadPool(cpu_count() - 1)
        res = pool.uimap(fit_subject, directories, chunksize=1000)

        fit_subject(directories[0])

        n_fitted = 0
        data_cache = list()

        for subject_data in res:
            n_fitted += 1
            progbar.update(n_fitted)
            if subject_data is not None:
                data_cache.extend(subject_data)
            if len(data_cache) % 100 == 0 and len(data_cache) != 0:
                self.partial_fit(np.concatenate(data_cache))
                data_cache = list()

        if len(data_cache) != 0:
            self.partial_fit(np.concatenate(data_cache))

        self.save()

    def fit_dataset(self, X, discretizer=None, imputer=None, verbose=True, save=True):
        """_summary_

        Args:
            discretizer (_type_): _description_
            X (_type_): _description_
        """
        if imputer is None:

            def preprocessor(frame):
                return discretizer.make_categorical_data(discretizer.make_imputed_data(frame))
        else:

            def preprocessor(frame):
                return imputer.transform(frame)

        if isinstance(X, np.ndarray):
            self.fit(preprocessor(X))
        else:
            start = time.time()
            [
                self.partial_fit(preprocessor(frame.reshape(1, -1)))
                for timeseries in X
                for frame in timeseries
            ]
            print(f"n_samples_seen_ {self.n_samples_seen_}")
            if verbose:
                info_io(f"Done computing new normalizer in {end-start}!")
            end = time.time()
        if save:
            self.save()


class MinMaxNormalizer(MinMaxScaler):
    """
    """

    def __init__(self, storage_path=None, verbose=1):
        """_summary_

        Args:
            storage_path (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 1.
        """
        self.storage_path = storage_path
        self.verbose = verbose
        super().__init__()

    def save(self):
        """_summary_
        """
        with open(self.storage_path, "wb") as save_file:
            pickle.dump(obj=self.__dict__, file=save_file, protocol=2)

    def load(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.storage_path.is_file():
            if os.path.getsize(self.storage_path) > 0:
                with open(self.storage_path, "rb") as load_file:
                    load_params = pickle.load(load_file)
        else:
            return 0

        for key, value in load_params.items():
            setattr(self, key, value)

        return 1

    def fit_dataset(self, discretizer, X):
        """_summary_

        Args:
            discretizer (_type_): _description_
            X (_type_): _description_
        """
        start = time.time()
        [
            self.partial_fit(discretizer.make_categorical_data(
                discretizer.make_imputed_data(frame))) for frame in X
        ]
        end = time.time()

        if self.storage_path:
            self.save()

        if self.verbose:
            info_io(f"Done computing new normalizer in {end-start}!")

    def fit_reader(self, reader, discretizer=None, imputer=None):
        """_summary_

        Args:
            discretizer (_type_): _description_
            reader (_type_): _description_
        """
        directories = [path for path in reader.root_path.iterdir() if path.name.isnumeric()]
        directories = directories[:int(len(directories))]

        progbar = Progbar(len(directories), unit_name='step')

        if imputer is None:

            def fit_subject(folder):
                data = reader.read_sample(folder)
                X_subjects, y_subject = data["X"], data["y"]

                if X_subjects is None or y_subject is None:
                    return None

                return [
                    discretizer.make_categorical_data(discretizer.make_imputed_data(frame.reshape(1, -1))) \
                    for frame in X_subjects
                ]
        else:

            def fit_subject(folder):
                data = reader.read_sample(folder)
                X_subjects, y_subject = data["X"], data["y"]

                if X_subjects is None or y_subject is None:
                    return None

                return [
                    imputer.transform(sample.reshape(1, -1))
                    for frame in X_subjects
                    for sample in frame
                ]

        pool = ThreadPool(cpu_count() - 1)
        res = pool.uimap(fit_subject, directories, chunksize=50)

        fit_subject(directories[0])

        n_fitted = 0
        data_cache = list()

        for subject_data in res:
            n_fitted += 1
            progbar.update(n_fitted)
            if subject_data is not None:
                data_cache.extend(subject_data)
            if len(data_cache) % 100 == 0 and len(data_cache) != 0:
                self.partial_fit(np.concatenate(data_cache))
                data_cache = list()

        if len(data_cache) != 0:
            self.partial_fit(np.concatenate(data_cache))

        self.save()
