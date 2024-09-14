"""
Dataset Reader Module
=====================

This module provides classes and methods for reading and handling dataset files related to medical data, 
specifically from the MIMIC-III dataset. These classes are designed to facilitate the extraction, processing, 
and management of large and complex datasets, enabling efficient data manipulation and analysis.

The following table describes the input and output for each reader in this module:

+----------------------+-------------------+----------------------------------+
| Reader               | Input             | Reads                            |
+======================+===================+==================================+
| ExtractedSetReader   | Extracted dataset | Timeseries, episodic data,       |
|                      |                   | subject events, diagnoses, ICU   |
|                      |                   | history                          |
+----------------------+-------------------+----------------------------------+
| ProcessedSetReader   | Processed dataset | Samples and individual subject   |
|                      |                   | data                             |
+----------------------+-------------------+----------------------------------+
| EventReader          | CHARTEVENTS,      | Event data in chunks or full     |
|                      | OUTPUTEVENTS,     | dataset                          |
|                      | LABEVENTS         |                                  |
+----------------------+-------------------+----------------------------------+
| SplitSetReader       | Dataset split     | Training, validation, and test   |
|                      | information       | sets                             |
+----------------------+-------------------+----------------------------------+

References
----------
- YerevaNN/mimic3-benchmarks: https://github.com/YerevaNN/mimic3-benchmarks
"""
import random
import re
import os
import threading
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp
from collections.abc import Iterable
from copy import deepcopy
from metrics import CustomBins, LogBins
from collections import defaultdict
from utils.IO import *
from settings import *
from utils.timeseries import read_timeseries, subjects_for_samples
from utils.arrays import get_iterable_dtype, isiterable, zeropad_samples
from utils.types import NoopLock
from .mimic_utils import upper_case_column_names, convert_dtype_dict, read_varmap_csv
from .trackers import ExtractionTracker, PreprocessingTracker
from typing import List, Union, Dict

__all__ = ["ExtractedSetReader", "ProcessedSetReader", "EventReader", "SplitSetReader"]


class AbstractReader(object):
    """
    A base reader class for datasets, providing methods to handle and sample subject directories.

    Parameters
    ----------
    root_path : Path
        The root directory path containing subject folders.
    subject_ids : list of int, optional
        List of subject IDs to read. If None, reads all subject directories in the root_path.

    Raises
    ------
    ValueError
        If the specified subject IDs do not have existing directories.

    Examples
    --------
    >>> root_path = Path("/path/to/data")
    >>> reader = AbstractReader(root_path, subject_ids=[10006, 10011, 10019])
    >>> reader.subject_ids
    [10006, 10011, 10019]
    """

    def __init__(self, root_path: Path, subject_ids: List[int] = None) -> None:
        self._root_path = (root_path if isinstance(root_path, Path) else Path(root_path))

        if subject_ids is None:
            self._subject_folders = [
                folder for folder in self._root_path.iterdir()
                if folder.is_dir() and folder.name.isnumeric()
            ]
            self._update_self = True
        elif not subject_ids:
            warn_io("List of subjects passed to mimic dataset reader is empty!")
            self._update_self = False
            self._subject_folders = []
        else:
            self._update_self = False
            if all([Path(str(folder)).is_dir() for folder in subject_ids]):
                self._subject_folders = subject_ids
            elif all([Path(self._root_path, str(folder)).is_dir() for folder in subject_ids]):
                self._subject_folders = [
                    Path(self._root_path, str(folder)) for folder in subject_ids
                ]
            else:
                raise ValueError(
                    f"The following subject do not have existing directories: "
                    f"{*[ Path(str(folder)).name for folder in subject_ids if not (Path(self._root_path, str(folder)).is_dir() or Path(str(folder)).is_dir())],}"
                )

    def _update(self):
        """
        Update the list of subject folders. Does not update if subject IDs were specified on creation.
        """
        # Doesn't update if subject_ids specified on creation
        if self._update_self:
            self._subject_folders = [
                folder for folder in self._root_path.iterdir()
                if folder.is_dir() and folder.name.isnumeric()
            ]

    def _cast_dir_path(self, dir_path: Union[Path, str, int]) -> Path:
        """
        Cast the directory path to a Path object and ensure it is relative to the root path.
        """
        if isinstance(dir_path, int):
            dir_path = Path(str(dir_path))
        elif isinstance(dir_path, str):
            dir_path = Path(dir_path)
        if not dir_path.is_relative_to(self._root_path):
            dir_path = Path(self._root_path, dir_path)
        return dir_path

    def _cast_subject_ids(self, subject_ids: Union[List[str], List[int], np.ndarray]) -> List[int]:
        """
        Cast the subject IDs to a list of integers.
        """
        if subject_ids is None:
            return None
        return [int(subject_id) for subject_id in subject_ids]

    def _sample_ids(self, subject_ids: list, num_subjects: int, seed: int = 42):
        """
        Sample a specified number of subject IDs.
        """
        # Subject ids overwrites num subjects
        random.seed(seed)
        self._update()
        if subject_ids is not None:
            return subject_ids
        if num_subjects is not None:
            random.seed(seed)
            return random.sample(self.subject_ids, num_subjects)
        return self._subject_folders

    @property
    def root_path(self) -> Path:
        """
        Get the root directory path.

        Returns
        -------
        Path
            The root directory path.

        Examples
        --------
        >>> reader.root_path
        PosixPath('/path/to/data')
        """
        return self._root_path

    @property
    def subject_ids(self) -> List[int]:
        """
        Get the list of subject IDs either past as parameter or located in the directory.

        Returns
        -------
        List[int]
            The list of subject IDs.

        Examples
        --------
        >>> reader.subject_ids
        [10006, 10011, 10019]
        """
        return [int(folder.name) for folder in self._subject_folders]

    def _init_returns(self, file_types: tuple, read_ids: bool = True):
        """
        Initialize a dictionary or list to store the data to be read, depending on read IDs.
        """
        return {file_type: {} if read_ids else [] for file_type in file_types}


class ExtractedSetReader(AbstractReader):
    """
    A reader for extracted datasets, providing methods to read various types of data including
    timeseries, episodic data, events, diagnoses, and ICU history.

    Examples
    --------
    >>> root_path = Path("/path/to/data")
    >>>
    >>> # Init reader and optionally limit to subjects
    >>> reader = ExtractedSetReader(root_path, subject_ids=[10006, 10011, 10019])
    >>> # Read the timeseries
    >>> timeseries = reader.read_timeseries(num_subjects=2)
    >>> # Read the episodic data
    >>> episodic_data = reader.read_episodic_data(subject_ids=[10006, 10011])
    >>> # Read the data for one subject
    >>> subject_data = reader.read_subject(dir_path="10019", read_ids=True)
    >>> subject_data["episodic_data"].columns.tolist()
        [Icustay,AGE,LOS,MORTALITY,GENDER,ETHNICITY, ... ,71590,2869,2763,5770,V5865,99662,28860]
    >>> list(subject_data.keys())
        ['timeseries', 'episodic_data', 'subject_events', 'subject_diagnoses', 'subject_icu_history']
    >>> # Read the data for several subjects
    >>> subjects_data = reader.read_subjects(subject_ids=[10006, 10011], read_ids=True)
    >>> subjects_data[10006]["episodic_data"].columns.tolist()
        [Icustay,AGE,LOS,MORTALITY,GENDER,ETHNICITY, ... ,71590,2869,2763,5770,V5865,99662,28860]
    >>> # Read multiple subjects as list
    >>> subjects_data = reader.read_subjects(num_subjects=2)
    >>> subjects_data[0]["timeseries"][0].columns.tolist()
        [Capillary refill rate,Diastolic blood pressure, ... Systolic blood pressure,Temperature]

    Parameters
    ----------
    root_path : Path
        The root directory path containing subject folders.
    subject_ids : list of int, optional
        List of subject IDs to read. If None, reads all subjects in the root_path.
    num_samples : int, optional
        Number of samples to read. If None, reads all available samples.
    """

    convert_datetime = ["INTIME", "CHARTTIME", "OUTTIME", "ADMITTIME", "DISCHTIME", "DEATHTIME"]

    def __init__(self, root_path: Path, subject_ids: list = None) -> None:
        """_summary_

        Args:
            root_path (Path): _description_
            subject_folders (list, optional): _description_. Defaults to None.
        """

        self._file_types = ("timeseries", "episodic_data", "subject_events", "subject_diagnoses",
                            "subject_icu_history")
        # Maps from file type to expected index name
        self._index_name_mapping = dict(zip(self._file_types[1:], ["Icustay", None, None, None]))
        # Maps from file type to dtypes
        self._dtypes = {
            file_type: DATASET_SETTINGS[file_index]["dtype"]
            for file_type, file_index in zip(self._file_types, [
                "timeseries",
                "episodic_data",
                "subject_events",
                "diagnosis",
                "icu_history",
            ])
        }
        self._convert_datetime = {
            "subject_icu_history": DATASET_SETTINGS["icu_history"]["convert_datetime"],
            "subject_events": DATASET_SETTINGS["subject_events"]["convert_datetime"]
        }
        super().__init__(root_path, subject_ids)

    def read_csv(self, path: Path, dtypes: tuple = None) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame, converting specified columns to datetime.

        Parameters
        ----------
        path : Path
            Absolute or relative path to the CSV file.
        dtypes : tuple, optional
            Data type(s) to apply to either the whole dataset or individual columns.

        Returns
        -------
        pd.DataFrame
            The dataframe read from the specified location.
        """
        file_path = Path(path)
        if not file_path.is_relative_to(self._root_path):
            file_path = Path(self._root_path, file_path)

        if not file_path.is_file():
            warn_io(f"File path {str(file_path)} does not exist!")
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path,
                             dtype=dtypes,
                             na_values=[''],
                             keep_default_na=False,
                             low_memory=False)
        except TypeError as error:
            error_io(f"Can't fit the integer range into requested dtype. Pandas error: {error}",
                     TypeError)

        df = upper_case_column_names(df)

        for column in set(df.columns) & set(self.convert_datetime):
            df[column] = pd.to_datetime(df[column], errors="coerce")

        return df

    def read_subject(self,
                     dir_path: Union[Path, int, str],
                     read_ids: bool = False,
                     file_type_keys: bool = True,
                     file_types: tuple = None):
        """
        Read data for a single subject for specified directory or subject ID.

        Parameters
        ----------
        dir_path : Union[Path, int, str]
            The directory path to read the subject data from.
        read_ids : bool, optional
            Whether to read IDs. Defaults to False.
        file_type_keys : bool, optional
            Whether to use file type keys in the returned dictionary. Defaults to True.
        file_types : tuple, optional
            The types of files to read. If None, reads all file types.

        Returns
        -------
        dict
            Dictionary containing the data read for the subject.
        """
        dir_path = self._cast_dir_path(dir_path)

        if file_types is None:
            file_types = self._file_types
        else:
            if not (isinstance(file_types, Iterable) and not isinstance(file_types, str)):
                raise ValueError(f'file_types must be a iterable but is {type(file_types)}')

        return_data = dict() if file_type_keys else list()

        if not self._check_subject_dir(dir_path,
                                       [file for file in file_types if not file == "timeseries"]):
            return {}

        for filename in file_types:
            if filename == "timeseries":
                if file_type_keys:
                    return_data["timeseries"] = self._get_timeseries(dir_path, read_ids)
                else:
                    return_data.append(self._get_timeseries(dir_path, read_ids))
            else:
                if file_type_keys:
                    return_data[filename] = self._read_file(filename, dir_path)
                else:
                    return_data.append(self._read_file(filename, dir_path))
        if not len(return_data):
            warn_io(f"Directory {str(dir_path)} does not exist!")
        return return_data

    def read_subjects(self,
                      subject_ids: Union[List[str], List[int], None] = None,
                      num_subjects: int = None,
                      read_ids: bool = False,
                      file_type_keys: bool = True,
                      seed: int = 42):
        """
        Read data for multiple subjects, with file keys being one of timeseries, episodic_data, subject_events,
        diagnosis or icu_history.

        Parameters
        ----------
        subject_ids : Union[List[str], List[int]], optional
            List of subject IDs to read. If None, reads all subjects.
        num_subjects : int, optional
            Number of subjects to read. If None, reads all available subjects.
        read_ids : bool, optional
            Whether to read IDs. Defaults to False.
        file_type_keys : bool, optional
            Whether to use file type keys in the returned dictionary. Defaults to True.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the data read for the subjects.
        """
        subject_ids = self._cast_subject_ids(subject_ids)

        if subject_ids is not None and num_subjects is not None:
            raise ValueError("Only one of subject_ids or num_subjects can be specified!")

        subject_ids = self._sample_ids(subject_ids, num_subjects, seed)

        if read_ids:
            return_data = dict()
            for subject_id in subject_ids:
                subject_path = Path(self._root_path, str(subject_id))
                if not subject_path.is_dir():
                    continue
                subject_id = int(subject_path.name)
                return_data[subject_id] = self.read_subject(dir_path=Path(subject_path),
                                                            file_type_keys=file_type_keys,
                                                            file_types=self._file_types,
                                                            read_ids=read_ids)
            assert all([len(subject) for subject in return_data.values()])
            return return_data

        # without ids
        return_data = list()
        for subject_path in subject_ids:
            return_data.append(
                self.read_subject(dir_path=Path(subject_path),
                                  file_types=self._file_types,
                                  file_type_keys=file_type_keys,
                                  read_ids=read_ids))
        return return_data

    def read_timeseries(self,
                        num_subjects: int = None,
                        subject_ids: int = None,
                        read_ids: bool = False,
                        seed: int = 42):
        """
        Read timeseries data for specified subjects.

        Parameters
        ----------
        num_subjects : int, optional
            Number of subjects to read. Default is None.
        subject_ids : int, optional
            List of subject IDs to read. Default is None.
        read_ids : bool, optional
            Whether to read IDs. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the timeseries data for the subjects.
        """
        return self._read_filetype("timeseries", num_subjects, subject_ids, read_ids, seed)

    def read_episodic_data(self,
                           num_subjects: int = None,
                           subject_ids: int = None,
                           read_ids: bool = False,
                           seed: int = 42):
        """
        Read episodic data for specified subjects.

        Parameters
        ----------
        num_subjects : int, optional
            Number of subjects to read. Default is None.
        subject_ids : int, optional
            List of subject IDs to read. Default is None.
        read_ids : bool, optional
            Whether to read IDs. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the episodic data for the subjects.
        """
        return self._read_filetype("episodic_data", num_subjects, subject_ids, read_ids, seed)

    def read_events(self,
                    num_subjects: int = None,
                    subject_ids: int = None,
                    read_ids: bool = False,
                    seed: int = 42):
        """
        Read event data for specified subjects.

        Parameters
        ----------
        num_subjects : int, optional
            Number of subjects to read. Default is None.
        subject_ids : int, optional
            List of subject IDs to read. Default is None.
        read_ids : bool, optional
            Whether to read IDs. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the event data for the subjects.
        """
        return self._read_filetype("subject_events", num_subjects, subject_ids, read_ids, seed)

    def read_diagnoses(self,
                       num_subjects: int = None,
                       subject_ids: int = None,
                       read_ids: bool = False,
                       seed: int = 42):
        """
        Read diagnosis data for specified subjects.

        Parameters
        ----------
        num_subjects : int, optional
            Number of subjects to read. Default is None.
        subject_ids : int, optional
            List of subject IDs to read. Default is None.
        read_ids : bool, optional
            Whether to read IDs. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the diagnosis data for the subjects.
        """
        return self._read_filetype("subject_diagnoses", num_subjects, subject_ids, read_ids, seed)

    def read_icu_history(self,
                         num_subjects: int = None,
                         subject_ids: int = None,
                         read_ids: bool = False,
                         seed: int = 42):
        """
        Read ICU history data for specified subjects.

        Parameters
        ----------
        num_subjects : int, optional
            Number of subjects to read. Default is None.
        subject_ids : int, optional
            List of subject IDs to read. Default is None.
        read_ids : bool, optional
            Whether to read IDs. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the ICU history data for the subjects.
        """
        return self._read_filetype("subject_icu_history", num_subjects, subject_ids, read_ids, seed)

    def _read_filetype(
        self,
        file_type: str,
        num_subjects: int,
        subject_ids: Union[List[int], List[str], np.ndarray],
        read_ids: bool,
        seed: int,
    ):
        """
        Reads the specified type of data file for multiple subjects.

        This method fetches the data for the given file type (e.g., timeseries, episodic_data) from the dataset 
        for either a specified list of subjects or a specified number of subjects. It ensures reproducibility 
        by using a random seed.
        """
        subject_ids = self._cast_subject_ids(subject_ids)

        if subject_ids is not None and num_subjects is not None:
            raise ValueError("Only one of subject_ids or num_subjects can be specified!")

        subject_ids = self._sample_ids(subject_ids, num_subjects, seed)
        subject_folders = [Path(self.root_path, str(subject_id)) for subject_id in subject_ids]
        if read_ids:
            return self._read_data_with_ids(subject_folders, file_type)
        return self._read_data_without_ids(subject_folders, file_type)

    def _read_data_with_ids(self, subject_folders: Path, file_type: str):
        """
        Reads data with subject IDs for multiple subjects.

        This method retrieves the data for a specified file type for multiple subjects, 
        including their IDs. It is particularly useful when the dataset includes IDs and the user 
        wants to maintain the association between data and subject IDs.
        """
        return_data = dict()
        for subject_path in subject_folders:
            subject_id = int(subject_path.name)
            return_data.update({
                subject_id:
                    self.read_subject(dir_path=Path(subject_path),
                                      file_types=[file_type],
                                      file_type_keys=False,
                                      read_ids=True).pop()
            })
        return return_data

    def _read_data_without_ids(self, subject_folders: Path, file_type: str):
        """
        Reads data without subject IDs for multiple subjects.

        This method retrieves the data for a specified file type for multiple subjects, 
        excluding their IDs. It is useful when IDs are not necessary, and the focus is 
        on the data itself.
        """
        return_data = list()
        for subject_id in subject_folders:
            if file_type == 'timeseries':
                return_data.extend(
                    self.read_subject(dir_path=Path(subject_id),
                                      file_types=[file_type],
                                      file_type_keys=False,
                                      read_ids=False).pop())
            else:
                return_data.append(
                    self.read_subject(dir_path=Path(subject_id),
                                      file_types=[file_type],
                                      file_type_keys=False,
                                      read_ids=False).pop())
        return return_data

    def _check_subject_dir(self, subject_folder: Path, file_types: tuple):
        """
        Checks if the subject directory contains the required files.

        This method verifies the presence of the necessary data files for the specified file types 
        in the subject's directory. It helps ensure that all required files are available before 
        attempting to read the data.
        """
        if os.getenv("DEBUG"):
            for filename in file_types:
                if not Path(subject_folder, f"{filename}.csv").is_file():
                    debug_io(f"Directory {subject_folder} does not have file {filename}.csv")
        return all([
            True if Path(subject_folder, f"{filename}.csv").is_file() else False
            for filename in file_types
        ])

    def _read_file(self, filename: str, dir_path: Path):  # , return_data: dict):
        """
        Reads a specific file from the directory.

        This method reads the data file for the given filename from the specified directory path. 
        It applies the necessary data types and converts date-time columns as required. This method 
        is a core part of the data extraction process for individual files.
        """
        file_df = pd.read_csv(Path(dir_path, f"{filename}.csv"),
                              dtype=self._dtypes[filename],
                              index_col=self._index_name_mapping[filename],
                              na_values=[''],
                              keep_default_na=False,
                              low_memory=False)

        if filename in self._convert_datetime:
            for column in self._convert_datetime[filename]:
                file_df[column] = pd.to_datetime(file_df[column])
        return file_df

    def _get_timeseries(self, dir_path: Path, read_ids: bool):
        """
        Retrieves timeseries data for a subject.

        This method reads and processes the timeseries data files from the specified directory path. 
        It can optionally include subject IDs in the returned data. This is useful for handling 
        timeseries data separately due to its unique, stay wise, structure.
        """
        subject_files = os.listdir(dir_path)
        if read_ids:
            timeseries = dict()
        else:
            timeseries = list()

        for file in subject_files:
            stay_id = re.findall('[0-9]+', file)
            if not stay_id:
                continue

            stay_id = stay_id.pop()
            if file.replace(stay_id, "") == "timeseries_.csv":
                if read_ids:
                    timeseries[int(stay_id)] = pd.read_csv(
                        Path(dir_path, file),
                        na_values=[''],
                        keep_default_na=False,
                        dtype=self._dtypes["timeseries"]).set_index('hours')
                else:
                    timeseries.append(
                        pd.read_csv(Path(dir_path, file),
                                    na_values=[''],
                                    keep_default_na=False,
                                    dtype=self._dtypes["timeseries"]).set_index('hours'))

        return timeseries


class ProcessedSetReader(AbstractReader):
    """
    A reader for processed datasets, providing methods to read samples and individual subject data.

    Examples
    --------
    >>> root_path = Path("/path/to/preprocessed/data")
    >>> reader = ProcessedSetReader(root_path, subject_ids=[10006, 10011, 10019])
    >>> # Reading two random samples as dictionary
    >>> X, y = reader.random_samples(n_samples=2, read_ids=True).values()
    >>> list(X.keys())
        [10006, 10019]
    >>> X[10006][244351].columns
        [Capillary refill rate,Diastolic blood pressure, ... Systolic blood pressure,Temperature,Weight]
    >>> # Reading two random samples as list
    >>> samples = reader.random_samples(n_samples=2)
    >>> samples["X"]
        [[Capillary refill rate  Diastolic blood pressure  ... Systolic blood pressure  Temperature  Weight
            [14425 rows x 17 columns],
         [Capillary refill rate  Diastolic blood pressure  ... Systolic blood pressure  Temperature  Weight
            [12025 rows x 17 columns]]
    >>> data, ids = reader.random_samples(n_samples=3, return_ids=True)
    >>> ids
    >>> [10006, 10011, 10019]
    >>> # Read specific samples
    >>> reader.read_samples(subject_ids=[10006, 10011], read_ids=True)
    >>> # Read single sample
    >>> reader.read_sample(10006, read_ids=True)

    Parameters
    ----------
    root_path : Path
        The root directory path containing subject folders.
    subject_ids : list of int, optional
        List of subject IDs to read. If None, reads all subjects in the root_path.
    set_index : bool, optional
        Whether to set the index for the dataframes. Defaults to True.
    """

    def __init__(self, root_path: Path, subject_ids: list = None) -> None:
        self._reader_switch = {
            "csv":
                defaultdict(lambda: self._read_csv, \
                {"X": (lambda x: self._read_csv(x, dtypes=DATASET_SETTINGS["timeseries"]["dtype"]))}),
            "npy":
                defaultdict(lambda: np.load),
            "h5":
                defaultdict(lambda: pd.read_hdf, {"X": self._read_hdf})
        }
        super().__init__(root_path, subject_ids)
        self._random_ids = deepcopy(self.subject_ids)
        self._convert_datetime = ["INTIME", "CHARTTIME", "OUTTIME"]
        self._possibgle_datatypes = [pd.DataFrame, np.ndarray, np.array, None]

    @staticmethod
    def _read_csv(path: Path, dtypes: tuple = None) -> pd.DataFrame:
        df = pd.read_csv(path, na_values=[''], keep_default_na=False, dtype=dtypes)
        if 'hours' in df.columns:
            df = df.set_index('hours')
        if 'Timestamp' in df.columns:
            df = df.set_index('Timestamp')
        if 'bins' in df.columns:
            df = df.set_index('bins')
        return df

    @staticmethod
    def _read_hdf(path: Path, dtypes: tuple = None) -> pd.DataFrame:
        df = pd.read_hdf(path)
        if 'hours' in df.columns:
            df = df.set_index('hours')
        if 'Timestamp' in df.columns:
            df = df.set_index('Timestamp')
        if 'bins' in df.columns:
            df = df.set_index('bins')
        return df

    def read_samples(self,
                     subject_ids: Union[List[str], List[int]] = None,
                     read_ids: bool = False,
                     read_timestamps: bool = False,
                     read_masks: bool = False,
                     data_type=None):
        """
        Read samples for the specified subject IDs, either as dictionary with ID keys or as list.

        Parameters
        ----------
        subject_ids : Union[List[str], List[int]], optional
            List of subject IDs to read. If None, reads all subjects.
        read_ids : bool, optional
            Whether to read IDs. Defaults to False.
        read_timestamps : bool, optional
            Whether to read timestamps. Defaults to False.
        data_type : type, optional
            Data type to cast the read data to. Can be one of [pd.DataFrame, np.ndarray, None]. Defaults to None.

        Returns
        -------
        dict
            Dictionary containing the samples read.
        """
        y_key = "yds" if read_masks else "y"
        dataset = {"X": {}, y_key: {}} if read_ids else {"X": [], y_key: []}

        if read_masks:
            dataset.update({"M": {} if read_ids else []})

        if read_timestamps:
            dataset.update({"t": {} if read_ids else []})

        if subject_ids is None:
            subject_ids = self.subject_ids

        subject_ids = self._cast_subject_ids(subject_ids)

        for subject_id in subject_ids:
            sample = self.read_sample(subject_id,
                                      read_ids=read_ids,
                                      read_timestamps=read_timestamps,
                                      read_masks=read_masks,
                                      data_type=data_type)
            for prefix in sample:
                if not len(sample[prefix]):
                    warn_io(f"Subject {subject_id} does not exist!")
                if read_ids:
                    dataset[prefix].update({subject_id: sample[prefix]})
                else:
                    dataset[prefix].extend(sample[prefix])

        return dataset

    def read_sample(self,
                    subject_id: Union[int, str],
                    read_ids: bool = False,
                    read_timestamps: bool = False,
                    read_masks: bool = False,
                    data_type=None) -> dict:
        """
        Read data for a single subject.

        Parameters
        ----------
        subject_id : Union[int, str]
            The subject ID to read.
        read_ids : bool, optional
            Whether to read IDs. Defaults to False.
        read_timestamps : bool, optional
            Whether to read timestamps. Defaults to False.
        data_type : type, optional
            Data type to cast the read data to. Can be one of [pd.DataFrame, np.ndarray, None]. Defaults to None.

        Returns
        -------
        dict
            Dictionary containing the data read for the subject.

        Raises
        ------
        ValueError
            If the data_type is not one of the possible data types.
        """
        y_key = "yds" if read_masks else "y"
        subject_id = int(subject_id)
        if not data_type in self._possibgle_datatypes:
            raise ValueError(
                f"Parameter data_type must be one of {self._possibgle_datatypes} but is {data_type}."
            )
        if data_type == np.array:
            data_type = np.ndarray

        dir_path = Path(self._root_path, str(subject_id))

        def _extract_number(string: str) -> int:
            stripper = f"abcdefghijklmnopqrstuvwxyzABZDEFGHIJKLMNOPQRSTUVWXYZ."
            return int(string.replace(".h5", "").replace(".csv", "").strip(stripper).strip("_"))

        def _convert_file_data(X):
            if data_type is None:
                return X
            if not isinstance(X, data_type) and data_type == np.ndarray:
                return X.to_numpy()
            elif not isinstance(X, data_type) and data_type == pd.DataFrame:
                return pd.DataFrame(X)
            return X

        dataset = {"X": {}, y_key: {}} if read_ids else {"X": [], y_key: []}

        if read_masks:
            dataset.update({"M": {} if read_ids else []})

        if read_timestamps:
            dataset.update({"t": {} if read_ids else []})

        stay_id_stack = list()
        for file in dir_path.iterdir():
            stay_id = _extract_number(file.name)
            file_extension = file.suffix.strip(".")
            reader = self._reader_switch[file_extension]
            reader_kwargs = ({"allow_pickle": True} if file_extension == "npy" else {})

            if stay_id in stay_id_stack:
                continue

            stay_id_stack.append(stay_id)
            for prefix in dataset.keys():
                file_path = Path(file.parent, f"{prefix}_{stay_id}{file.suffix}")
                if not file_path.is_file():
                    continue
                file_data = reader[prefix](file_path, **reader_kwargs)
                file_data = _convert_file_data(file_data)

                if prefix == "t" and not read_timestamps:
                    continue

                if prefix == "M" and not read_masks:
                    continue

                if read_ids:
                    if subject_id not in dataset[prefix]:
                        dataset[prefix].update({_extract_number(file.name): file_data})
                    else:
                        dataset[prefix][_extract_number(file.name)] = file_data
                else:
                    dataset[prefix].append(file_data)

        return dataset

    def random_samples(
            self,
            n_subjects: int = 1,
            read_ids: bool = False,
            read_timestamps: bool = False,
            data_type=None,
            return_ids: bool = False,  # This is for debugging
            read_masks: bool = False,
            seed: int = 42):
        """
        Sample subjects randomly without replacement until subject list is exhauasted.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to read. Default is 1.
        read_ids : bool, optional
            Whether to read IDs. Default is False.
        read_timestamps : bool, optional
            Whether to read timestamps. Default is False.
        data_type : type, optional
            Data type to cast the read data to. Can be one of [pd.DataFrame, np.ndarray, None]. Default is None.
        return_ids : bool, optional
            Whether to return the sampled IDs along with the data. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the sampled data.
        list, optional
            List of sampled subject IDs if return_ids is True.
        """
        random.seed(seed)
        sample_ids = list()
        n_samples_needed = n_subjects

        while n_samples_needed > 0:
            if not self._random_ids:
                self._random_ids = list(set(self.subject_ids) - set(sample_ids))
                random.shuffle(self._random_ids)

            n_samples_curr = min(len(self._random_ids), n_samples_needed)
            sample_ids.extend(self._random_ids[:n_samples_curr])
            self._random_ids = self._random_ids[n_samples_curr:]
            n_samples_needed -= n_samples_curr

            if len(sample_ids) >= len(self.subject_ids):
                if len(sample_ids) > len(self.subject_ids):
                    warn_io(
                        f"Maximum number of samples in dataset reached! Requested {n_subjects}, but dataset size is {len(self.subject_ids)}."
                    )
                break
        if return_ids:
            return self.read_samples(sample_ids,
                                     read_ids=read_ids,
                                     read_timestamps=read_timestamps,
                                     read_masks=read_masks,
                                     data_type=data_type), sample_ids
        return self.read_samples(sample_ids,
                                 read_ids=read_ids,
                                 read_timestamps=read_timestamps,
                                 read_masks=read_masks,
                                 data_type=data_type)

    def to_numpy(self,
                 n_samples: int = None,
                 scaler=None,
                 imputer=None,
                 subject_ids: Union[List[str], List[int]] = None,
                 deep_supervision: bool = False,
                 normalize_inputs: bool = False,
                 read_timestamps: bool = False,
                 data_type=None,
                 bining: str = "none",
                 one_hot: bool = False,
                 return_ids: bool = False,
                 seed: int = 42):
        """
        Convert the dataset to a NumPy array of dim (#ofSamples, maxTimeSteps, Features).

        This function reads the specified number of samples or samples of specified subject IDs from the dataset,
        applies optional scaling and imputation, and returns the data in NumPy array format. It can also
        return the IDs of the subjects if specified. The function only works if the dataset is entirely numeric,
        that is only after categorization has been applied. (Discretization or Feature engineering)

        Parameters
        ----------
        n_samples : int, optional
            The number of samples to read. If `subject_ids` is specified, this parameter is ignored. Default is None.
        scaler : object, optional
            An object that implements the `transform` method, used to scale the data. Default is None.
        imputer : object, optional
            An object that implements the `transform` method, used to impute missing values in the data. Default is None.
        subject_ids : list of int or list of str, optional
            A list of subject IDs to read. If specified, `n_samples` is ignored. Default is None.
        deep_supervision : bool, optional
            Whether to read the dataset in deep supervision mode. If True, returns prefices X, M, and yds where M is the masks
            and yds is the deep supervision targets. Default is False.
        normalize_inputs : bool, optional
            If True, ensures that the time step dimension of the targets equals the time dimension of the samples.
            A mask is returned indicating where the original targets are located in the series. Default is False.
        read_timestamps : bool, optional
            Whether to read timestamps. Default is False.
        data_type : type, optional
            The type to cast the read data to. Can be one of [pd.DataFrame, np.ndarray, None]. Default is None.
        return_ids : bool, optional
            Whether to return the IDs of the subjects along with the data. Default is False.
        seed : int, optional
            Random seed for reproducibility when sampling. Default is 42.

        Returns
        -------
        dict
            A dictionary containing the dataset. Keys include 'X' for the data, 'y' for the labels, and optionally
            'M' for masks, 't' for timestamps if `read_timestamps` is True, and 'yds' for deep supervision targets
            if `deep_supervision` is True.
        list of int or list of str, optional
            A list of subject IDs if `return_ids` is True.

        Raises
        ------
        ValueError
            If `data_type` is not one of the possible data types (pd.DataFrame, np.ndarray, None).
        """
        # TODO! fix num samples
        if one_hot and bining == "none":
            warn_io("One hot encoding is specified but no bining is applied."
                    " Ignoring one hot encoding.")

        if subject_ids:
            # read only specified subject ids
            if n_samples:
                warn_io("Both n_samples and subject_ids are specified. Ignoring n_samples.")

            dataset = self.read_samples(subject_ids,
                                        read_timestamps=read_timestamps,
                                        read_masks=deep_supervision,
                                        data_type=data_type)

        elif n_samples:
            tracker = PreprocessingTracker(storage_path=Path(self._root_path, "progress"))
            subject_ids, _ = subjects_for_samples(tracker,
                                                  target_size=n_samples,
                                                  deep_supervision=deep_supervision)
            dataset = self.read_samples(subject_ids,
                                        read_timestamps=read_timestamps,
                                        read_masks=deep_supervision,
                                        data_type=data_type)

        else:
            # read all episodes limited by n_samples
            dataset, subject_ids = self.random_samples(n_subjects=len(self.subject_ids),
                                                       read_timestamps=read_timestamps,
                                                       data_type=data_type,
                                                       return_ids=True,
                                                       read_masks=deep_supervision,
                                                       seed=seed)

        prefices = deepcopy(list(dataset.keys()))

        if deep_supervision:
            if bining == "custom":
                dataset["yds"] = [
                    CustomBins.get_bin_custom(x, one_hot=one_hot).reshape(*x.shape)
                    for x in dataset["yds"]
                ]
            elif bining == "log":
                dataset["yds"] = [
                    LogBins.get_bin_log(x, one_hot=one_hot).reshape(*x.shape)
                    for x in dataset["yds"]
                ]
        else:
            # Buffer dataset to allow for iteration
            buffer_dataset = dict(zip(prefices, [[] for _ in range(len(prefices))]))
            sample_count = 0
            for idx in range(len(dataset["X"])):
                X_df, y_df = dataset["X"][idx], dataset["y"][idx]
                X_dfs, y_dfs, ts = read_timeseries(X_df,
                                                   y_df,
                                                   bining=bining,
                                                   one_hot=one_hot,
                                                   dtype=pd.DataFrame)

                # Add samples to buffer
                buffer_dataset["X"].extend(X_dfs)
                buffer_dataset["y"].extend(y_dfs)
                sample_count += len(y_dfs)

            dataset = buffer_dataset
            del buffer_dataset

        # Normalize lengths on the smallest times stamp
        if normalize_inputs:
            for idx in range(n_samples):
                length = min([int(dataset[prefix][idx].index[-1]) \
                            for prefix in prefices])
                for prefix in prefices:
                    dataset[prefix][idx] = dataset[prefix][idx][:length]

            # Needs masking if a series
            if not "M" in dataset:
                dataset["M"] = list()
                for idx in range(n_samples):
                    y_reindex_df = dataset["y"][idx].reindex(dataset["X"][idx].index)
                    dataset["y"][idx] = y_reindex_df.fillna(0)
                    dataset["M"].append((~y_reindex_df.isna()).astype(int))

        # Apply transformations on load (impute, scale) if specified
        if imputer is not None:
            dataset["X"] = [imputer.transform(sample) for sample in dataset["X"]]
        if scaler is not None:
            dataset["X"] = [scaler.transform(sample) for sample in dataset["X"]]
        if scaler is None and imputer is None:
            dataset["X"] = [sample.values for sample in dataset["X"]]

        # Zeropad and concat the dataset
        for prefix in deepcopy(list(dataset.keys())):
            if len(dataset[prefix]) \
               and isiterable(dataset[prefix][0]) \
               and len(dataset[prefix][0].shape) > 1:
                dataset[prefix] = zeropad_samples(dataset[prefix])
            elif len(dataset[prefix]) and isiterable(dataset[prefix][0]):
                dataset[prefix] = np.stack(dataset[prefix])
                dataset[prefix] = np.expand_dims(dataset[prefix], 1)
            else:
                dataset[prefix] = np.array(dataset[prefix],
                                           dtype=get_iterable_dtype(dataset[prefix])).reshape(
                                               -1, 1, 1)
        if return_ids:
            return dataset, subject_ids
        return dataset


class EventReader():
    """
    A reader for event data from CHARTEVENTS, OUTPUTEVENTS, and LABEVENTS, providing methods 
    to read data either in chunks or in a single shot.

    This class is designed to facilitate the extraction and manipulation of event data from the MIMIC-III dataset. 
    It supports reading data in manageable chunks for efficient processing and allows filtering by specific subject IDs.

    Examples
    --------
    Initialize an EventReader with a specific dataset folder and subject IDs, and read data in chunks:

    >>> from pathlib import Path
    >>> dataset_folder = Path("/path/to/dataset")
    >>> event_reader = EventReader(dataset_folder, subject_ids=[10006, 10011, 10019], chunksize=1000)
    >>> event_frames, frame_lengths = event_reader.get_chunk()
    >>> event_frames["CHARTEVENTS.csv"].columns.tolist()
        [SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUEUOM']
    >>> event_frames["LABEVENTS.csv"].columns.tolist()
        [SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUEUOM']
    >>> event_frames["OUTPUTEVENTS.csv"].columns.tolist()
        [SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUEUOM']
    >>> # Or read all event data at once:
    >>> event_reader = EventReader(dataset_folder)
    >>> all_events = event_reader.get_all()
    >>> all_events.columns.tolist()
        [SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUEUOM']	

    Parameters
    ----------
    dataset_folder : Path
        The path to the dataset folder containing event data files.
    subject_ids : list of int, optional
        List of subject IDs to read. If None, reads all subjects. Defaults to None.
    chunksize : int, optional
        The size of chunks to read at a time. Defaults to None, which means the entire file is read at once.
    tracker : ExtractionTracker, optional
        An object to track the extraction progress. Defaults to None.
    """

    def __init__(self,
                 dataset_folder: Path,
                 subject_ids: list = None,
                 chunksize: int = None,
                 tracker: ExtractionTracker = None,
                 verbose: bool = True,
                 lock: mp.Lock = NoopLock()) -> None:

        self.dataset_folder = dataset_folder
        self._done = False
        self._lock = lock
        self._verbose = verbose

        # Logic to early terminate if none of the subjects are in the remaining dataset
        if subject_ids is not None and len(subject_ids):
            self._last_occurrence = {
                "CHARTEVENTS.csv": {},
                "LABEVENTS.csv": {},
                "OUTPUTEVENTS.csv": {}
            }
            subject_target = dict(
                zip([str(subject_id) for subject_id in subject_ids],
                    ["SUBJECT_ID"] * len(subject_ids)))
            self._subject_ids = [int(subject_id) for subject_id in subject_ids]
            self._lo_thread_response = {}  # Last occurence thread response
            self._lo_thread_done = {}
            for csv in self._last_occurrence:
                self._lo_thread_response[csv] = threading.Event()
                thread = threading.Thread(target=self._csv_find_last,
                                          args=(Path(self.dataset_folder, csv), subject_target,
                                                self._last_occurrence[csv],
                                                self._lo_thread_response[csv]))
                thread.start()
                self._lo_thread_done[csv] = False

        else:
            self._lo_thread_response = None
            self._lo_thread_done = {
                "CHARTEVENTS.csv": False,
                "LABEVENTS.csv": False,
                "OUTPUTEVENTS.csv": False
            }
            self._subject_ids = None

        self._chunksize = chunksize
        self._tracker = tracker
        self._csv_settings = DATASET_SETTINGS["CHARTEVENTS"]
        self._event_csv_kwargs = {
            "CHARTEVENTS.csv": {
                "dtype": convert_dtype_dict(DATASET_SETTINGS["CHARTEVENTS"]["dtype"])
            },
            "LABEVENTS.csv": {
                "dtype": convert_dtype_dict(DATASET_SETTINGS["LABEVENTS"]["dtype"])
            },
            "OUTPUTEVENTS.csv": {
                "dtype": convert_dtype_dict(DATASET_SETTINGS["OUTPUTEVENTS"]["dtype"])
            }
        }
        self.event_csv_skip_rows = dict(zip(self._event_csv_kwargs.keys(), [0, 0, 0]))

        if chunksize:
            # Readers from which to get chunk
            # Pandas is leaky when get chunk is not carried out to the EOF so we
            # have to manage the file handle ourselves
            self._csv_reader = dict()
            self._csv_handle = dict()
            for csv_name, kwargs in self._event_csv_kwargs.items():
                file_handle = Path(dataset_folder, csv_name).open("rb")
                self._csv_reader[csv_name] = pd.read_csv(file_handle,
                                                         iterator=True,
                                                         na_values=[''],
                                                         keep_default_na=False,
                                                         chunksize=chunksize,
                                                         low_memory=False,
                                                         **kwargs)
                self._csv_handle[csv_name] = file_handle
            if subject_ids is None:
                # If subject ids is specified we need to start from the begining again
                # Since the new subjects might be in the first chunks
                self._init_reader()

        self._convert_datetime = ["INTIME", "CHARTTIME", "OUTTIME"]
        resource_folder = Path(dataset_folder, "resources")
        assert resource_folder.is_dir(), FileNotFoundError(
            f"Folder {str(resource_folder)} does not contain resources folder, which in turn should contain"
            f" hcup_ccs_2015_definitions.yaml and itemid_to_variable_map.csv.")
        self._varmap_df = read_varmap_csv(resource_folder)

    @property
    def done_reading(self):
        """
        Check if all chunks have been read.

        Returns
        -------
        bool
            True if all chunks have been read, False otherwise.
        """
        return self._done

    def _init_reader(self):
        """
        Initialize the reader by skipping rows according to the tracker if it exists.
        """
        info_io(f"Starting reader initialization.", verbose=self._verbose)
        header = "Initializing reader and starting at row:\n"
        msg = list()
        for csv in self._event_csv_kwargs:
            try:
                with self._lock:
                    n_chunks = self._tracker.count_subject_events[csv] // self._chunksize
                    skip_rows = self._tracker.count_subject_events[csv] % self._chunksize
                [len(self._csv_reader[csv].get_chunk()) for _ in range(n_chunks)]  # skipping chunks
                self.event_csv_skip_rows[csv] = skip_rows  # rows to skip in first chunk
                with self._lock:
                    msg.append(f"{csv}: {self._tracker.count_subject_events[csv]}")
            except:
                self._csv_handle[csv].close()
        info_io(header + " - ".join(msg), verbose=self._verbose)

    def get_chunk(self) -> tuple:
        """
        Get the next chunk of event data and the lengths of the returned frames with keys CHARTEVENTS.csv, LABEVENTS.csv, and OUTPUTEVENTS.csv.

        Returns
        -------
        tuple
            A tuple containing:
            - event_frames: dict of pd.DataFrame
                A dictionary where keys are CSV file names and values are dataframes of the read chunk.
            - frame_lengths: dict of int
                A dictionary where keys are CSV file names and values are the number of events read in the chunk.        
        """

        def read_frame(csv_name):
            # Read a frame
            if self._lo_thread_response is not None:
                if not self._lo_thread_done[csv_name] and self._lo_thread_response[csv_name].is_set(
                ):
                    last_occurences = max(self._last_occurrence[csv_name].values())
                    self._last_occurrence[csv_name] = last_occurences
                    debug_io(f"Last occurence for {csv_name}: {last_occurences}")
                    self._lo_thread_done[csv_name] = True

            events_df = self._csv_reader[csv_name].get_chunk()

            # If start index exceeds last occurence of any subject, stop reader
            if self._lo_thread_done[csv_name] and events_df.index[0] >= self._last_occurrence[
                    csv_name]:
                debug_io(
                    f"Reader for {csv_name} is done on last occurence at line {events_df.index[0]}."
                )
                self._csv_handle[csv_name].close()
                if all([handle.closed for handle in self._csv_handle.values()]):
                    self._done = True

            # Uppercase column names for consistency
            events_df = upper_case_column_names(events_df)

            if self._subject_ids is not None:
                events_df = events_df[events_df["SUBJECT_ID"].isin(self._subject_ids)]

            if not 'ICUSTAY_ID' in events_df:
                events_df['ICUSTAY_ID'] = pd.NA
                events_df['ICUSTAY_ID'] = events_df['ICUSTAY_ID'].astype(
                    self._event_csv_kwargs[csv_name]["dtype"]["ICUSTAY_ID"])

            # Drop specified columns and NAN rows, merge onto varmap for variable definitions
            drop_cols = set(events_df.columns) - set(self._csv_settings["columns"])
            events_df = events_df.drop(drop_cols, axis=1)

            # Skip start rows if directory already existed
            if self.event_csv_skip_rows[csv_name]:
                events_df = events_df.iloc[self.event_csv_skip_rows[csv_name]:]
                self.event_csv_skip_rows[csv_name] = 0

            # Convert to datetime
            for column in self._csv_settings["convert_datetime"]:
                events_df[column] = pd.to_datetime(events_df[column])
            if not events_df.empty and self._lo_thread_response is not None:
                debug_io(
                    f"Csv: {csv_name}\nRead chunk of size: {len(events_df)}\nLast idx: {events_df.index[-1]}"
                )

            return events_df

        event_frames = dict()

        for csv_name in self._event_csv_kwargs:
            # Read frame, if reader is done, it will raise
            try:
                if not self._csv_handle[csv_name].closed:
                    event_frames[csv_name] = read_frame(csv_name)
            except StopIteration as error:
                debug_io(f"Reader finished on {error}")
                self._csv_handle[csv_name].close()
                if all([handle.closed for handle in self._csv_handle.values()]):
                    self._done = True

            # Readers are done, return empty frame if queried
            if all([handle.closed for handle in self._csv_handle.values()]):
                return {csv_name: pd.DataFrame() for csv_name in self._event_csv_kwargs}, dict()

        # Number of subject events per CVS type
        frame_lengths = {csv_name: 0 for csv_name in self._event_csv_kwargs}
        frame_lengths.update({csv_name: len(frame) for csv_name, frame in event_frames.items()})

        return event_frames, frame_lengths

    def get_all(self):
        """
        Get all event data from the dataset.

        Returns
        -------
        pd.DataFrame
            Dataframe containing all event data.

        """
        event_csv = ["CHARTEVENTS.csv", "LABEVENTS.csv", "OUTPUTEVENTS.csv"]
        event_frames = list()

        for csv in event_csv:
            events_df = pd.read_csv(Path(self.dataset_folder, csv),
                                    low_memory=False,
                                    na_values=[''],
                                    keep_default_na=False,
                                    **self._event_csv_kwargs[csv])
            events_df = upper_case_column_names(events_df)

            if not 'ICUSTAY_ID' in events_df:
                events_df['ICUSTAY_ID'] = np.nan
                events_df['ICUSTAY_ID'] = events_df['ICUSTAY_ID'].astype(pd.Int32Dtype())

            # Drop specified columns and NAN rows, merge onto varmap for variable definitions
            drop_cols = set(events_df.columns) - set(self._csv_settings["columns"])
            events_df = events_df.drop(drop_cols, axis=1)

            for column in self._csv_settings["convert_datetime"]:
                events_df[column] = pd.to_datetime(events_df[column])

            event_frames.append(events_df)

        full_df = pd.concat(event_frames, ignore_index=True)

        return full_df

    @staticmethod
    def _csv_find_last(csv_path: Path,
                       target_dict: dict,
                       last_occurences: dict,
                       done_event: threading.Event = None):
        """
        Finds the last line where the values in the target_dict are found in the csv file.
        Target_dict should be a dictionary with the target value as key and the target column name as value.
        """
        with open(csv_path, 'r', encoding='utf-8') as file:
            # Check if the given column name is valid
            headers = file.readline().strip().split(',')
            column_indices = {name.strip("\'\""): index for index, name in enumerate(headers)}

            missing_columns = [
                column_name for column_name in target_dict.values()
                if column_name not in column_indices
            ]
            if missing_columns:
                raise ValueError(f"Column name '{*missing_columns,}' does not exist in the file")

            # Column->idx and idx->column mapping
            column_indices = {
                column_name: column_indices[column_name] for column_name in target_dict.values()
            }
            value_to_idx = {value: column_indices[target_dict[value]] for value in target_dict}
            # Init counts
            line_number = 0
            last_occurences.update(dict(zip(target_dict.keys(), [0] * len(target_dict))))

            columns_of_interest = list(set(value_to_idx.values()))
            max_column_of_interest = max(columns_of_interest) + 1

            # Run through the file
            for line in file:
                columns = line.strip().split(',', maxsplit=max_column_of_interest)
                # Check if the target column has the target value
                line_number += 1
                for target_value in last_occurences.keys():
                    if target_value == columns[value_to_idx[target_value]]:
                        last_occurences[target_value] = line_number

        done_event.set()
        debug_io(
            f"Thread for {csv_path.name} has and found last occurence {max(last_occurences.values())}."
        )
        return last_occurences


class SplitSetReader(object):
    """
    A reader for datasets split into training, validation, and test sets, providing access to each split.

    Examples
    --------
    >>> root_path = Path("/path/to/data")
    >>> split_sets = {
    ...     "train": [1, 2, 3],
    ...     "val": [4, 5],
    ...     "test": [6, 7]
    ... }
    >>> split_reader = SplitSetReader(root_path, split_sets)
    >>> split_reader.split_names
        ['train', 'val', 'test']
    >>> split_reader.root_path
        PosixPath('/path/to/data')
    >>> train_reader = split_reader.train
    >>> train_dataset = train_reader.read_samples()
    >>> val_reader = split_reader.val
    >>> test_reader = split_reader.test

    Parameters
    ----------
    root_path : Path
        The root directory path containing the dataset.
    split_sets : dict of {str: list of int}
        A dictionary where keys are split names (e.g., "train", "val", "test") and values are lists of subject IDs.
    """

    def __init__(self, root_path: Path, split_sets: Dict[str, List[int]]) -> None:
        self._root_path = Path(root_path)
        self._subject_ids = split_sets
        self._readers = {
            split: ProcessedSetReader(self._root_path, subject_ids=split_sets[split])
            for split in split_sets
            if split_sets[split]
        }

        self._splits = list(self._readers.keys())
        cum_length = sum([len(split) for split in split_sets.values()])
        self._ratios = {split: len(split_sets[split]) / cum_length for split in split_sets}

    @property
    def split_names(self) -> list:
        """
        Get the names of the dataset splits.

        Returns
        -------
        list
            List of split names.
        """
        return self._splits

    @property
    def root_path(self):
        """
        Get the root path.

        Returns
        -------
        Path
            The root directory path.
        """
        return self._root_path

    @property
    def train(self) -> ProcessedSetReader:
        """
        Get the reader for the training set.

        Returns
        -------
        ProcessedSetReader
            The reader for the training set, or None if not available.
        """
        if "train" in self._readers:
            return self._readers["train"]
        return

    @property
    def val(self) -> ProcessedSetReader:
        """
        Get the reader for the validation set.

        Returns
        -------
        ProcessedSetReader
            The reader for the validation set, or None if not available.
        """
        if "val" in self._readers:
            return self._readers["val"]
        return

    @property
    def test(self) -> ProcessedSetReader:
        """
        Get the reader for the test set.

        Returns
        -------
        ProcessedSetReader
            The reader for the test set, or None if not available.
        """
        if "test" in self._readers:
            return self._readers["test"]
        return
