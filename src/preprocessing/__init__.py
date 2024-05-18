import os
import numpy as np
import pickle
from utils.IO import *
from tensorflow.keras.utils import Progbar
from datasets.readers import ProcessedSetReader
from pathlib import Path
from typing import List, Tuple
from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """_summary_
    """

    @abstractmethod
    def __init__(self) -> None:
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        ...

    @property
    @abstractmethod
    def subjects(self) -> List[int]:
        ...

    @abstractmethod
    def transform(self, *args, **kwargs):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        ...

    @abstractmethod
    def transform_subject(self, subject_id: int) -> Tuple[dict, dict, dict]:
        ...

    @abstractmethod
    def save_data(self, subject_ids: list = None) -> None:
        ...


class AbstractScikitProcessor(ABC):

    @abstractmethod
    def __init__(self, storage_path: Path):
        """_summary_

        Args:
            storage_path (_type_): _description_
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray):
        ...

    @abstractmethod
    def fit(self, X: np.ndarray):
        ...

    @abstractmethod
    def partial_fit(self, X: np.ndarray):
        ...

    def save(self, storage_path=None):
        """_summary_
        """
        if storage_path is not None:
            self._storage_path = Path(storage_path, "scaler.pkl")
        if self._storage_path is None:
            raise ValueError("No storage path provided!")
        with open(self._storage_path, "wb") as save_file:
            pickle.dump(obj=self.__dict__, file=save_file, protocol=2)

    def load(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self._storage_path is not None:
            if self._storage_path.is_file():
                if os.path.getsize(self._storage_path) > 0:
                    with open(self._storage_path, "rb") as load_file:
                        load_params = pickle.load(load_file)
                    for key, value in load_params.items():
                        setattr(self, key, value)

                    return 1
        return 0

    def fit_dataset(self, X):
        """_summary_

        Args:
            discretizer (_type_): _description_
            X (_type_): _description_
        """
        if self._verbose:
            info_io(f"Fitting scaler to dataset of size {len(X)}")
            progbar = Progbar(len(X), unit_name='step')
        n_fitted = 0

        for frame in X:
            if hasattr(self, "_imputer") and self._imputer is not None:
                frame = self._imputer.transform(frame)
            self.partial_fit(frame)
            n_fitted += 1
            if self._verbose:
                progbar.update(n_fitted)

        if self._storage_path:
            self.save()

        if self._verbose:
            info_io(f"Done computing new normalizer.")
        return self

    def fit_reader(self, reader: ProcessedSetReader):
        """_summary_

        Args:
            discretizer (_type_): _description_
            reader (_type_): _description_
        """
        if self._verbose:
            info_io(f"Fitting scaler to reader of size {len(reader.subject_ids)}")
            progbar = Progbar(len(reader.subject_ids), unit_name='step')

        n_fitted = 0

        for subject_id in reader.subject_ids:
            X_subjects, _ = reader.read_sample(subject_id).values()
            for frame in X_subjects:
                if hasattr(self, "_imputer") and self._imputer is not None:
                    frame = self._imputer.transform(frame)
                self.partial_fit(frame)
            n_fitted += 1
            if self._verbose:
                progbar.update(n_fitted)
        if self._storage_path is None:
            self.save(reader.root_path)
        else:
            self.save()

        if self._verbose:
            info_io(f"Done computing new normalizer.\nSaved in location {self._storage_path}!")

        return self
