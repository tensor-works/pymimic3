import time
import pickle
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from utils.IO import *
from tensorflow.keras.utils import Progbar
from pathos.multiprocessing import cpu_count
from pathos.pools import ThreadPool


class BatchImputer(SimpleImputer):

    def __init__(self,
                 missing_values=np.nan,
                 strategy='mean',
                 verbose=0,
                 copy=True,
                 storage_path=None):
        """_summary_

        Args:
            storage_path (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 1.
        """
        self.storage_path = storage_path
        self.verbose = verbose
        self.statistics_ = None
        self.n_features_in_ = 0
        self.n_samples_in_ = 0
        super().__init__(missing_values=missing_values, strategy=strategy, copy=copy)

    def save(self):
        """_summary_
        """
        self.statistics_ = np.nan_to_num(self.statistics_)
        with open(self.storage_path, "wb") as save_file:
            pickle.dump(obj=self.__dict__, file=save_file, protocol=2)

        return

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

    def fit_dataset(self, X):
        """_summary_

        Args:
            discretizer (_type_): _description_
            X (_type_): _description_
        """
        if isinstance(X, np.ndarray):
            self.fit(X)
            if self.verbose:
                info_io(f"Done computing new partial scaler!")
        else:
            start = time.time()
            [self.partial_fit(frame) for frame in X]
            end = time.time()
            if self.verbose:
                info_io(f"Done computing new partial scaler in {end-start}!")

        if self.storage_path:
            self.save()

        return

    def fit_reader(self, reader):
        """_summary_

        Args:
            discretizer (_type_): _description_
            reader (_type_): _description_
        """
        warnings.filterwarnings("error")
        directories = [path for path in reader.root_path.iterdir() if path.name.isnumeric()]
        directories = directories[:int(len(directories))]

        progbar = Progbar(len(directories), unit_name='step')

        def fit_subject(folder):
            data = reader.read_sample(folder)
            X_subjects, y_subject = data["X"], data["y"]

            if X_subjects is None or y_subject is None:
                return None

            return [frame for frame in X_subjects]

        pool = ThreadPool(cpu_count() - 1)
        res = pool.uimap(fit_subject, directories, chunksize=50)

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

        if len(data_cache):
            self.partial_fit(np.concatenate(data_cache))

        self.save()

    def partial_fit(self, X):
        """_summary_

        Args:
            X (_type_): _description_
        """
        warnings.filterwarnings("ignore")
        n = len(X)
        if np.isnan(X).all():
            return
        avg = np.nanmean(X, axis=0)

        if self.statistics_ is None:
            self.fit(X)
            self.n_samples_in_ = n
        else:
            self.statistics_ = np.nanmean([self.n_samples_in_ * self.statistics_, n * avg],
                                          axis=0) / (self.n_samples_in_ + n)
        self.n_samples_in_ += n
        warnings.filterwarnings("default")
        return
