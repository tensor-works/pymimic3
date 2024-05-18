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
import pandas as pd
import shutil
import numpy as np
from pathos.helpers import mp
import operator
from pathlib import Path
from utils.IO import *
from functools import reduce

__all__ = ["DataSetWriter"]


class DataSetWriter():
    """_summary_
    """

    def __init__(self, root_path: Path) -> None:
        """_summary_

        Args:
            root_path (Path): _description_
        """
        self.root_path = root_path

    def check_filename(self, filename: str):
        """_summary_

        Args:
            filename (str): _description_
        """
        possible_filenames = [
            "episodic_data", "timeseries", "subject_events", "subject_diagnoses",
            "subject_icu_history", "X", "y", "t", "header"
        ]

        if filename not in possible_filenames:
            raise (f"choose a filename from {possible_filenames}")

    def get_subject_ids(self, data: dict):
        """_summary_

        Args:
            data (dict): _description_

        Returns:
            _type_: _description_
        """

        id_sets = [set(dictionary.keys()) for dictionary in data.values()]
        subject_ids = list(reduce(operator.and_, id_sets))
        return subject_ids

    def write_bysubject(self,
                        data: dict,
                        index: bool = True,
                        exists_ok: bool = False,
                        file_type: str = "csv"):
        """_summary_

        Args:
            data (dict): _description_
        """
        if self.root_path is None:
            return

        if not file_type in ["csv", "npy", "hdf5"]:
            raise ValueError(
                f"file_type {file_type} not supported. Must be one of ['csv', 'npy', 'hdf5']")

        for subject_id in self.get_subject_ids(data):

            self.write_file(subject_id=subject_id,
                            data={filename: data[filename][subject_id] for filename in data.keys()},
                            index=index,
                            exists_ok=exists_ok,
                            file_type=file_type)

        return

    def write_file(self,
                   subject_id: int,
                   data: dict,
                   index: bool = True,
                   exists_ok: bool = False,
                   file_type: str = "csv"):
        """_summary_

        Args:
            subject_id (int): _description_
            data (dict): _description_
            exists_ok (bool): switch to append mode if file exists for CSVs
        """

        def save_df(df: pd.DataFrame,
                    path: Path,
                    index: str = True,
                    file_type: str = "csv",
                    exists_ok: bool = False) -> None:
            if exists_ok and path.is_file() and not file_type == "hd5f":
                mode = "a"
                header = False
            else:
                mode = "w"
                header = True
            if file_type == "hdf5":
                pd.DataFrame(df).to_hdf(Path(path.parent, f"{path.stem}.h5"),
                                        key="data",
                                        mode=mode,
                                        index=index)
            elif file_type == "csv":
                pd.DataFrame(df).to_csv(Path(path.parent, f"{path.stem}.csv"),
                                        mode=mode,
                                        index=index,
                                        header=header)
            elif file_type == "npy":
                if isinstance(df, (pd.DataFrame, pd.Series)):
                    df = df.to_numpy()
                np.save(Path(path.parent, f"{path.stem}.npy"), df)

        if file_type in ["npy", "hdf5"] and exists_ok:
            raise ValueError("Append mode not supported for numpy files!")

        if not file_type in ["csv", "npy", "hdf5"]:
            raise ValueError(
                f"file_type {file_type} not supported. Must be one of ['csv', 'npy', 'hdf5']")
        for filename, item in data.items():
            delet_flag = False
            self.check_filename(filename)

            subject_path = Path(self.root_path, str(subject_id))

            if not subject_path.is_dir():
                subject_path.mkdir(parents=True, exist_ok=True)
            if isinstance(item, (pd.DataFrame, pd.Series, np.ndarray)):
                if not len(item):
                    continue
                csv_path = Path(subject_path, f"{filename}")
                save_df(df=item,
                        path=csv_path,
                        index=index,
                        file_type=file_type,
                        exists_ok=exists_ok)
            elif isinstance(item, dict):
                for icustay_id, data in item.items():
                    if not len(data):
                        continue
                    csv_path = Path(subject_path, f"{filename}_{icustay_id}")
                    save_df(df=data,
                            path=csv_path,
                            index=index,
                            file_type=file_type,
                            exists_ok=exists_ok)

            # do not create empty or incomplete folders
            if not [folder for folder in subject_path.iterdir()] or delet_flag:
                debug_io(
                    f"Removing folder {subject_path}, because a file is missing or the folder is empty!"
                )
                shutil.rmtree(str(subject_path))

    def write_subject_events(self, data: dict, lock: mp.Lock = None, dtypes: dict = None):
        """_summary_

        Args:
            root_path (pd.DataFrame): _description_
            data (dict): _description_
        """
        if self.root_path is None:
            return

        def write_csv(dataframe: pd.DataFrame, path: Path, lock: mp.Lock):
            if dataframe.empty:
                return
            if lock is not None:
                lock.acquire()
            if dtypes is not None:
                dataframe = dataframe.astype(dtypes)
            if not path.is_file():
                dataframe.to_csv(path, index=False)
            else:
                dataframe.to_csv(path, mode='a', index=False, header=False)

            if lock is not None:
                lock.release()
            return

        for subject_id, subject_data in data.items():
            subject_path = Path(self.root_path, str(subject_id))
            subject_path.mkdir(parents=True, exist_ok=True)
            subject_event_path = Path(subject_path, "subject_events.csv")

            write_csv(subject_data, subject_event_path, lock)

        return
