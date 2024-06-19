import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Union, Dict
from utils.IO import *
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm  # Importing tqdm for progress bars
from datasets.readers import ProcessedSetReader
from datasets.writers import DataSetWriter


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
        self._name: str = ...
        self._action: str = ...
        self._verbose: bool = ...
        self._storage_name: str = ...
        self._imputer: AbstractScikitProcessor = ...

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
            Path(storage_path).mkdir(parents=True, exist_ok=True)
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

    def fit_dataset(self, \
                    dataset: Dict[str, Union[Dict[str, Dict[str, Union[pd.DataFrame,np.ndarray]]], \
                                             List[Union[pd.DataFrame, np.ndarray]]]]):
        """
        Fit the processor to an entire dataset.

        Parameters
        ----------
        dataset : iterable
            The dataset to fit. The "X" key should indicate the features. The dataset can either be
            in list or dictionary fromat.

        Returns
        -------
        self
            The fitted processor.
        """
        X = dataset["X"]
        if self._verbose:
            info_io(f"Fitting {self._name} to dataset of size {len(X)}")
        n_fitted = 0
        with tqdm(total=len(X), unit='step', ascii=' >=', ncols=120,
                  disable=self._verbose) as progbar:
            if isinstance(X, list):
                for frame in X:
                    if hasattr(self, "_imputer") and self._imputer is not None:
                        frame = self._imputer.transform(frame)
                    self.partial_fit(frame)
                    n_fitted += 1
                    if self._verbose:
                        progbar.update(1)
            elif isinstance(X, dict):
                for subject_id, frames in X.items():
                    for stay_id, frame in frames.items():
                        if hasattr(self, "_imputer") and self._imputer is not None:
                            frame = self._imputer.transform(frame)
                        self.partial_fit(frame)
                    n_fitted += 1
                    if self._verbose:
                        progbar.update(1)
            else:
                raise ValueError(f"Unrecognized dictionary type {type(X)}. Should be list or dict")
        self.save()

        if self._verbose:
            info_io(f"Done computing new {self._name}.")
        return self

    def fit_reader(self, reader: ProcessedSetReader, save=False):
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
            info_io(f"Fitting {self._name} to reader of size {len(reader.subject_ids)}")

        with tqdm(total=len(reader.subject_ids),
                  unit='step',
                  ascii=' >=',
                  ncols=120,
                  disable=self._verbose) as progbar:
            n_fitted = 0

            for subject_id in reader.subject_ids:
                X_subjects, _ = reader.read_sample(subject_id).values()
                for frame in X_subjects:
                    if hasattr(self, "_imputer") and self._imputer is not None:
                        frame = self._imputer.transform(frame)
                    self.partial_fit(frame)
                n_fitted += 1
                if self._verbose:
                    progbar.update(1)

        self.save()

        if self._verbose:
            info_io(
                f"Done computing new {Path(self._storage_path).stem}.\nSaved in location {self._storage_path}!"
            )

        return self

    def transform_dataset(
        self,
        dataset: Dict[str, Union[Dict[str, Dict[str, Union[pd.DataFrame, np.ndarray]]], \
                                 List[Union[pd.DataFrame, np.ndarray]]]],
    ):
        X = dataset["X"]
        if self._verbose:
            info_io(f"{self._action.capitalize()} dataset of size {len(X)}")
        n_transformed = 0

        with tqdm(total=len(X), unit='step', ascii=' >=', ncols=120,
                  disable=self._verbose) as progbar:
            if isinstance(X, list):
                X_return = list()
                for frame in X:
                    X_return.append(self.transform(frame))
                    n_transformed += 1
                    if self._verbose:
                        progbar.update(1)
            elif isinstance(X, dict):
                X_return = dict()
                for subject_id, frames in X.items():
                    X_return[subject_id] = dict()
                    for stay_id, frame in frames.items():
                        X_return[subject_id].update({stay_id: self.transform(frame)})
                    n_transformed += 1
                    if self._verbose:
                        progbar.update(1)
            else:
                raise ValueError(f"Unrecognized dictionary type {type(X)}. Should be list or dict")
        dataset["X"] = X_return
        if self._verbose:
            info_io(f"Done transforming dataset using {self._name}.")
        return dataset

    def transform_reader(self, reader: ProcessedSetReader):
        if self._verbose:
            info_io(f"{self._action.capitalize()} reader with {len(reader.subject_ids)} samples.")
        dataset_writer = DataSetWriter(reader.root_path)
        with tqdm(total=len(reader.subject_ids),
                  unit='step',
                  ascii=' >=',
                  ncols=120,
                  disable=self._verbose) as progbar:
            n_transformed = 0

            for subject_id in reader.subject_ids:
                X_subjects, _ = reader.read_sample(subject_id, read_ids=True).values()
                X_transformed = dict()
                for stay_id, frame in X_subjects.items():
                    X_transformed.update({stay_id: self.transform(frame)})

                dataset_writer.write_bysubject({"X": {
                    subject_id: X_transformed
                }},
                                               file_type="dynamic")
                n_transformed += 1
                if self._verbose:
                    progbar.update(1)

        self.save()

        if self._verbose:
            info_io(f"Done transforming reader using {self._name}.\n"
                    f"Saved in location {reader.root_path}!")

        return reader

    def fit_transform_dataset(
        self,
        dataset: Dict[str, Union[Dict[str, Dict[str, Union[pd.DataFrame, np.ndarray]]],
                                 List[Union[pd.DataFrame, np.ndarray]]]],
    ):
        return self.fit_dataset(dataset).transform_dataset(dataset)

    def fit_transform_reader(self, reader: ProcessedSetReader):
        return self.fit_reader(reader).transform_reader(reader)
