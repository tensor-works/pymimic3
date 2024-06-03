import os
import numpy as np
import pickle
from utils.IO import *
from tensorflow.keras.utils import Progbar
from pathlib import Path
from abc import ABC, abstractmethod



class AbstractScikitProcessor(ABC):
    """
    Abstract base class for scikit-learn style processors.

    This class provides a template for processors that need to implement `transform`, `fit`, and `partial_fit`
    methods. It also includes methods for saving and loading the processor's state.

    Parameters
    ----------
    storage_path : Path
        The path where the processor's state will be stored.
    """

    @abstractmethod
    def __init__(self, storage_path: Path):
        """_summary_

        Args:
            storage_path (_type_): _description_
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray):
        """
        Transform the input data once the preprocessor has been fitted.

        Parameters
        ----------
        X : np.ndarray
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        ...

    @abstractmethod
    def fit(self, X: np.ndarray):
        """
        Fit the processor to the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit.
        """
        ...

    @abstractmethod
    def partial_fit(self, X: np.ndarray):
        """
        Partially fit the processor to the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data to partially fit.
        """
        ...

    def save(self, storage_path=None):
        """
        Save the processor's state to the storage path.

        Parameters
        ----------
        storage_path : Path, optional
            The path where the processor's state will be saved. If None, the existing storage path is used.

        Raises
        ------
        ValueError
            If no storage path is provided.
        """
        if storage_path is not None:
            self._storage_path = Path(storage_path, self._storage_name)
        if self._storage_path is None:
            raise ValueError("No storage path provided!")
        with open(self._storage_path, "wb") as save_file:
            pickle.dump(obj=self.__dict__, file=save_file, protocol=2)

    def load(self):
        """
        Load the processor's state from the storage path.

        Returns
        -------
        int
            1 if the state was successfully loaded, 0 otherwise.
        """
        if self._storage_path is not None:
            if self._storage_path.is_file():
                if os.path.getsize(self._storage_path) > 0:
                    info_io(f"Loading {Path(self._storage_path).stem} from:\n{self._storage_path}")
                    with open(self._storage_path, "rb") as load_file:
                        load_params = pickle.load(load_file)
                    for key, value in load_params.items():
                        setattr(self, key, value)

                    return 1
        return 0

    def fit_dataset(self, X):
        """
        Fit the processor to an entire dataset.

        Parameters
        ----------
        X : iterable
            The dataset to fit.

        Returns
        -------
        self
            The fitted processor.
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

        self.save()

        if self._verbose:
            info_io(f"Done computing new {Path(self._storage_path).stem}.")
        return self

    def fit_reader(self, reader, save=False):
        """
        Fit the processor to a dataset read from a reader.

        Parameters
        ----------
        reader : ProcessedSetReader
            The reader to read the dataset from.
        save : bool, optional
            Whether to save the processor's state after fitting, by default False.

        Returns
        -------
        self
            The fitted processor.
        """
        if self._storage_path is None:
            self._storage_path = Path(reader.root_path, self._storage_name)
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self.load():
            return self
        if self._verbose:
            info_io(
                f"Fitting {Path(self._storage_path).stem} to reader of size {len(reader.subject_ids)}"
            )
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

        self.save()

        if self._verbose:
            info_io(
                f"Done computing new {Path(self._storage_path).stem}.\nSaved in location {self._storage_path}!"
            )

        return self
