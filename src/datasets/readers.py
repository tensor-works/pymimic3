"""Dataset file

This file allows access to the dataset as specified.
All function in this file are used by the main interface function load_data.
Subfunctions used within private functions are located in the datasets.utils module.

Todo:
    - Use a settings.json
    - This is a construction site, see what you can bring in here
    - Provid link to kaggle in load_data doc string
    - Expand function to utils

YerevaNN/mimic3-benchmarks
"""
import random
import re
import os
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Iterable
from copy import deepcopy
from utils.IO import *
from settings import *
from .mimic_utils import upper_case_column_names, convert_dtype_dict, read_varmap_csv
from .trackers import ExtractionTracker
from typing import List, Union

__all__ = ["ExtractedSetReader", "ProcessedSetReader", "EventReader"]


class AbstractReader(object):

    def __init__(self, root_path: Path, subject_ids: list = None) -> None:
        """_summary_

        Args:
            root_path (Path): _description_
            subject_folders (list, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
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
        # Doesn't update if subject_ids specified on creation
        if self._update_self:
            self._subject_folders = [
                folder for folder in self._root_path.iterdir()
                if folder.is_dir() and folder.name.isnumeric()
            ]

    def _cast_dir_path(self, dir_path: Union[Path, str, int]) -> Path:
        if isinstance(dir_path, int):
            dir_path = Path(str(dir_path))
        elif isinstance(dir_path, str):
            dir_path = Path(dir_path)
        if not dir_path.is_relative_to(self._root_path):
            dir_path = Path(self._root_path, dir_path)
        return dir_path

    def _cast_subject_ids(self, subject_ids: Union[List[str], List[int], np.ndarray]) -> List[int]:
        if subject_ids is None:
            return None
        return [int(subject_id) for subject_id in subject_ids]

    def _sample_ids(self, subject_ids: list, num_subjects: int, seed: int = 42):
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
        return self._root_path

    @property
    def subject_ids(self) -> List[int]:
        """
        """
        return [int(folder.name) for folder in self._subject_folders]

    def _init_returns(self, file_types: tuple, read_ids: bool = True):
        """_summary_

        Args:
            file_types (tuple): _description_
        """
        return {file_type: {} if read_ids else [] for file_type in file_types}


class ExtractedSetReader(AbstractReader):

    convert_datetime = ["INTIME", "CHARTTIME", "OUTTIME", "ADMITTIME", "DISCHTIME", "DEATHTIME"]

    def __init__(self, root_path: Path, subject_ids: list = None, num_samples: int = None) -> None:
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
        """_summary_

        Args:
            path (Path): Absolute or relative path to csv.
            dtypes (tuple, optional): Data type(s) to apply to either the whole dataset or individual columns. 
                E.g., {'a': np.float64, 'b': np.int32, 'c': 'Int64'} Use str or object together with suitable 
                na_values settings to preserve and not interpret dtype. If converters are specified, they will be 
                applied INSTEAD of dtype conversion. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe from location
        """
        file_path = Path(path)
        if not file_path.is_relative_to(self._root_path):
            file_path = Path(self._root_path, file_path)

        if not file_path.is_file():
            warn_io(f"File path {str(file_path)} does not exist!")
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
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
        """_summary_

        Args:
            file_types (tuple): _description_
            dir_path (Path): _description_
            subject_id (int): _description_
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
        """_summary_

        Args:
            root_path (Path): _description_
            file_types (tuple, optional): _description_. Defaults to ("episodic_data", "subject_events", "subject_diagnoses", "subject_icu_history", "timeseries").
            num_subjects (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
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
        return self._read_filetype("timeseries", num_subjects, subject_ids, read_ids, seed)

    def read_episodic_data(self,
                           num_subjects: int = None,
                           subject_ids: int = None,
                           read_ids: bool = False,
                           seed: int = 42):
        return self._read_filetype("episodic_data", num_subjects, subject_ids, read_ids, seed)

    def read_events(self,
                    num_subjects: int = None,
                    subject_ids: int = None,
                    read_ids: bool = False,
                    seed: int = 42):
        return self._read_filetype("subject_events", num_subjects, subject_ids, read_ids, seed)

    def read_diagnoses(self,
                       num_subjects: int = None,
                       subject_ids: int = None,
                       read_ids: bool = False,
                       seed: int = 42):
        return self._read_filetype("subject_diagnoses", num_subjects, subject_ids, read_ids, seed)

    def read_icu_history(self,
                         num_subjects: int = None,
                         subject_ids: int = None,
                         read_ids: bool = False,
                         seed: int = 42):
        return self._read_filetype("subject_icu_history", num_subjects, subject_ids, read_ids, seed)

    def _read_filetype(
        self,
        file_type: str,
        num_subjects: int,
        subject_ids: Union[List[int], List[str], np.ndarray],
        read_ids: bool,
        seed: int,
    ):

        subject_ids = self._cast_subject_ids(subject_ids)

        if subject_ids is not None and num_subjects is not None:
            raise ValueError("Only one of subject_ids or num_subjects can be specified!")

        subject_ids = self._sample_ids(subject_ids, num_subjects, seed)
        subject_folders = [Path(self.root_path, str(subject_id)) for subject_id in subject_ids]
        if read_ids:
            return self._read_data_with_ids(subject_folders, file_type)
        return self._read_data_without_ids(subject_folders, file_type)

    def _read_data_with_ids(self, subject_folders: Path, file_type: str):
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
        """_summary_

        Args:
            dir_path (Path): _description_
            file_types (tuple): _description_

        Returns:
            _type_: _description_
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
        """_summary_

        Args:
            filename (str): _description_
            dir_path (Path): _description_
            subject_id (int): _description_
        """
        file_df = pd.read_csv(Path(dir_path, f"{filename}.csv"),
                              dtype=self._dtypes[filename],
                              index_col=self._index_name_mapping[filename],
                              low_memory=False)

        if filename in self._convert_datetime:
            for column in self._convert_datetime[filename]:
                file_df[column] = pd.to_datetime(file_df[column])
        return file_df

    def _get_timeseries(self, dir_path: Path, read_ids: bool):
        """_summary_

        Args:
            dir_path (Path): _description_
            subject_id (int): _description_
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
                        Path(dir_path, file), dtype=self._dtypes["timeseries"]).set_index('hours')
                else:
                    timeseries.append(
                        pd.read_csv(Path(dir_path, file),
                                    dtype=self._dtypes["timeseries"]).set_index('hours'))

        return timeseries


class ProcessedSetReader(AbstractReader):
    """_summary_
    """

    def __init__(self, root_path: Path, subject_ids: list = None, set_index: bool = True) -> None:
        """_summary_

        Args:
            root_path (Path): _description_
            subject_folders (list, optional): _description_. Defaults to None.
        """
        self._reader_switch_Xy = {
            "csv": {
                "X":
                    (lambda x: pd.read_csv(x, dtype=DATASET_SETTINGS["timeseries"]["dtype"]
                                          ).set_index('hours')
                     if set_index else pd.read_csv(x, dtype=DATASET_SETTINGS["timeseries"]["dtype"])
                    ),
                "y":
                    lambda x: pd.read_csv(x).set_index("Timestamp")
                    if "Timestamp" else pd.read_csv(x)
            },
            "npy": {
                "X": np.load,
                "y": np.load,
                "t": np.load
            }
        }
        super().__init__(root_path, subject_ids)
        self._random_ids = deepcopy(self.subject_ids)
        self._convert_datetime = ["INTIME", "CHARTTIME", "OUTTIME"]
        self._possibgle_datatypes = [pd.DataFrame, np.ndarray, np.array, None]

    def read_samples(self,
                     subject_ids: Union[List[str], List[int]] = None,
                     read_ids: bool = False,
                     read_timestamps: bool = False,
                     data_type=None):
        """_summary_

        Args:
            folder_names (List[str]): _description_
            read_stay_ids (bool, optional): _description_. Defaults to False.
            read_timestamps (bool, optional): _description_. Defaults to False.
        """

        dataset = {"X": {}, "y": {}} if read_ids else {"X": [], "y": []}

        if read_timestamps:
            dataset.update({"t": {} if read_ids else []})

        if subject_ids is None:
            subject_ids = self.subject_ids

        subject_ids = self._cast_subject_ids(subject_ids)

        for subject_id in subject_ids:
            sample = self.read_sample(subject_id,
                                      read_ids=read_ids,
                                      read_timestamps=read_timestamps,
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
                    data_type=None) -> dict:
        """_summary_

        Args:
            folder (Path): _description_
            folder_name (bool, optional): _description_. Defaults to False.
            read_timestamps (bool, optional): _description_. Defaults to False.
            data_type (_type_, optional): _description_. Defaults to None.
        """
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
            return int(string.strip(stripper).strip("_"))

        def _convert_file_data(X):
            if data_type is None:
                return X
            if not isinstance(X, data_type) and data_type == np.ndarray:
                return X.to_numpy()
            elif not isinstance(X, data_type) and data_type == pd.DataFrame:
                return pd.DataFrame(X)
            return X

        dataset = {"X": {}, "y": {}} if read_ids else {"X": [], "y": []}

        if read_timestamps:
            dataset.update({"t": {} if read_ids else []})

        stay_id_stack = list()
        for file in dir_path.iterdir():
            stay_id = _extract_number(file.name)
            file_extension = file.suffix.strip(".")
            reader = self._reader_switch_Xy[file_extension]
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

                if prefix == "t" and read_timestamps:
                    continue

                if read_ids:
                    if subject_id not in dataset[prefix]:
                        dataset[prefix].update({_extract_number(file.name): file_data})
                    else:
                        dataset[prefix][_extract_number(file.name)] = file_data
                else:
                    dataset[prefix].append(file_data)

        return dataset

    def random_samples(self,
                       n_samples: int = 1,
                       read_ids: bool = False,
                       read_timestamps: bool = False,
                       data_type=None,
                       seed: int = 42):
        """ Sampling without replacement from subjects

        Args:
            seed (_type_, optional): _description_. Defaults to 42:int.
        """
        random.seed(seed)
        sample_ids = list()
        n_samples_needed = n_samples

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
                        f"Maximum number of samples in dataset reached! Requested {n_samples}, but dataset size is {len(self.subject_ids)}."
                    )
                break

        return self.read_samples(sample_ids,
                                 read_ids=read_ids,
                                 read_timestamps=read_timestamps,
                                 data_type=data_type)


class EventReader():
    """_summary_
    """

    def __init__(self,
                 dataset_folder: Path,
                 subject_ids: list = None,
                 chunksize: int = None,
                 tracker: ExtractionTracker = None) -> None:
        """_summary_

        Args:
            dataset_folder (Path): _description_
            chunksize (int, optional): _description_. Defaults to None.
            tracker (object, optional): _description_. Defaults to None.
        """
        self.dataset_folder = dataset_folder
        self._done = False

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
        """Indicates wether all chunks have been read
        """
        return self._done

    def _init_reader(self):
        """_summary_
        """
        info_io(f"Starting reader initialization.")
        header = "Initializing reader and starting at row:\n"
        msg = list()
        for csv in self._event_csv_kwargs:
            try:
                n_chunks = self._tracker.count_subject_events[csv] // self._chunksize
                skip_rows = self._tracker.count_subject_events[csv] % self._chunksize
                [len(self._csv_reader[csv].get_chunk()) for _ in range(n_chunks)]  # skipping chunks
                self.event_csv_skip_rows[csv] = skip_rows  # rows to skip in first chunk
                msg.append(f"{csv}: {self._tracker.count_subject_events[csv]}")
            except:
                self._csv_handle[csv].close()
        info_io(header + " - ".join(msg))

    def get_chunk(self) -> tuple:
        """_summary_

        Returns:
            tuple: _description_
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
            if self._lo_thread_done[
                    csv_name] and events_df.index[0] >= self._last_occurrence[csv_name]:
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
        Returns:
            events_df:        Chartevent data from ICU bed
        """
        event_csv = ["CHARTEVENTS.csv", "LABEVENTS.csv", "OUTPUTEVENTS.csv"]
        event_frames = list()

        for csv in event_csv:
            events_df = pd.read_csv(Path(self.dataset_folder, csv),
                                    low_memory=False,
                                    **self._event_csv_kwargs[csv])
            events_df = upper_case_column_names(events_df)

            if not 'ICUSTAY_ID' in events_df:
                events_df['ICUSTAY_ID'] = pd.NA

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
        """Finds the last line where the values in the target_dict are found in the csv file.
        Target_dict should be a dictionary with the target value as key and the target column name as value.

        Args:
            csv_path (Path): _description_
            target_dict (dict): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
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
