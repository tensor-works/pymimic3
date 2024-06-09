"""
Processors Module
=================

This package provides classes for data preprocessing the MIMIC-III dataset. The processors transform
the data from the extracted format to a format suitable for different machine learning algorithms.
The data creation can be configured and the processors are created modularly easy modification.

Input and Output Description
----------------------------
+-------------------+-------------------+------------------------------------------+
| Processor         | Input             | Output                                   |
+===================+===================+==========================================+
| Preprocessor      | Extracted dataset | Task-specific processed data             |
|                   |                   | (output of data extraction)              |
+-------------------+-------------------+------------------------------------------+
| Discretizer       | Processed data    | Binned, one-hot encoded, and imputed     |
|                   |                   | data for use with neural networks        |
|                   |                   |                                          |
+-------------------+-------------------+------------------------------------------+
| Feature Engine    | Processed data    | Subsampled data engineered into a        |
|                   |                   | 714-length vector for classical machine  |
|                   |                   | learning algorithms                      |
+-------------------+-------------------+------------------------------------------+


"""

import random
import ray
import os
import pandas as pd
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Tuple, Union
from utils import dict_subset
from pathlib import Path
from abc import ABC, abstractmethod
from pathos.multiprocessing import cpu_count, Pool
from datasets.readers import ExtractedSetReader, ProcessedSetReader
from datasets.mimic_utils import copy_subject_info
from datasets.trackers import PreprocessingTracker
from utils.IO import *


class AbstractProcessor(ABC):
    """
    Abstract base class for data preprocessing in the MIMIC-III dataset.

    This class provides an interface for the processors, which transform datasets, transform individual 
    subjects and allow to save them to specified location. It ensures consistent structure 
    and processing steps for different preprocessing implementations in the MIMIC-III pipeline.
    """

    @abstractmethod
    def __init__(self) -> None:
        self._storage_path: Path = ...
        self._tracker: PreprocessingTracker = ...
        self._operation_name: str = ...
        self._task: str = ...
        self._verbose: bool = ...
        self._source_reader: Union[ExtractedSetReader, ProcessedSetReader] = ...

    @property
    @abstractmethod
    def subjects(self) -> List[int]:
        """
        Returns a list of subject IDs.

        Returns
        -------
        List[int]
            List of subject IDs.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        ...

    @abstractmethod
    def transform(self, *args, **kwargs):
        ...

    @abstractmethod
    def transform_subject(self, subject_id: int) -> Tuple[dict, dict, dict]:
        """
        Transforms data for a single subject, when a reader was passed at initialization.

        Parameters
        ----------
        subject_id : int
            ID of the subject to transform.

        Returns
        -------
        Tuple[dict, dict, dict]
            Transformed data for the subject.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        ...

    @abstractmethod
    def save_data(self, subject_ids: list = None) -> None:
        """
        Saves the processed data, either for all subjects if subject_ids = None, or for specified subjects.

        Parameters
        ----------
        subject_ids : list, optional
            List of subject IDs whose data should be saved. Defaults to None.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        ...

    def transform_dataset(self,
                          dataset: dict,
                          subject_ids: list = None,
                          num_subjects: int = None,
                          source_path: Path = None,
                          storage_path: Path = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Transforms and processes the dataset.

        This method processes the entire dataset, transforming the data for each subject, and then 
        saving the processed data to the specified location. It handles subject selection and ensures 
        the processed data is correctly stored.

        Parameters
        ----------
        dataset : dict
            The dataset to process.
        subject_ids : list, optional
            List of subject IDs to process. Defaults to None.
        num_subjects : int, optional
            Number of subjects to process. Defaults to None.
        source_path : Path, optional
            Source path of the data. Defaults to None.

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Processed data with keys 'X' for features and 'y' for labels.
        """
        if storage_path is not None:
            self._storage_path = storage_path
        copy_subject_info(source_path, self._storage_path)

        if self._tracker.is_finished:
            info_io(
                f"Compact {self._operation_name} already finalized in directory:\n{str(self._storage_path)}"
            )
            if num_subjects is not None:
                subject_ids = random.sample(self._tracker.subject_ids, k=num_subjects)
            return ProcessedSetReader(root_path=self._storage_path,
                                      subject_ids=subject_ids).read_samples(read_ids=True)

        info_io(f"Compact {self._operation_name}: {self._task}", level=0)

        subject_ids, excluded_subject_ids = self._get_subject_ids(num_subjects=num_subjects,
                                                                  subject_ids=subject_ids,
                                                                  all_subjects=dataset.keys())
        assert all([len(subject) for subject in dataset.values()])
        missing_subjects = 0
        if num_subjects is not None:
            X_subjects = dict()
            y_subjects = dict()
            while not len(X_subjects) == num_subjects:
                curr_dataset = dict_subset(dataset, subject_ids)
                X, y = self.transform(dataset=curr_dataset)
                X_subjects.update(X)
                y_subjects.update(y)
                it_missing_subjects = set(X.keys()) - set(subject_ids)
                subject_ids, excluded_subject_ids = self.get_subject_ids(
                    num_subjects=num_subjects - len(X_subjects),
                    subject_ids=None,
                    all_subjects=excluded_subject_ids)
                if it_missing_subjects:
                    missing_subjects += len(it_missing_subjects)
                    debug_io(f"Missing subjects are: {*it_missing_subjects,}")
                if not subject_ids:
                    break
                if len(X_subjects) == num_subjects:
                    debug_io(f"Missing {len(X_subjects) - num_subjects} subjects.")
                    debug_io(f"Unprocessable subjects are: {*it_missing_subjects,}")

        else:
            assert all([len(subject) for subject in dataset.values()])
            dataset = dict_subset(dataset, subject_ids)
            assert all([len(subject) for subject in dataset.values()])
            (X_subjects, y_subjects) = self.transform(dataset=dataset)
        if self._storage_path is not None:
            self.save_data()
            info_io(
                f"Finalized {self._operation_name} for {self._task} in directory:\n{str(self._storage_path)}"
            )
        else:
            info_io(f"Finalized {self._operation_name} for {self._task}.")
        self._tracker.is_finished = True
        return {"X": X_subjects, "y": y_subjects}

    def transform_reader(self,
                         reader: Union[ExtractedSetReader, ProcessedSetReader],
                         subject_ids: list = None,
                         num_subjects: int = None) -> ProcessedSetReader:
        """
        Transforms data using an extracted set reader.

        This method processes the data read by an ExtractedSetReader, transforming it for each subject,
        and then saving the processed data. It handles subject selection and ensures the processed data 
        is correctly stored.

        Parameters
        ----------
        reader : ExtractedSetReader
            Reader for the extracted set.
        subject_ids : list, optional
            List of subject IDs to process. Defaults to None.
        num_subjects : int, optional
            Number of subjects to process. Defaults to None.

        Returns
        -------
        ProcessedSetReader
            Reader for the processed set.
        """
        orig_verbose = self._verbose
        self._verbose = False
        self._source_reader = reader
        original_subject_ids = deepcopy(subject_ids)

        copy_subject_info(reader.root_path, self._storage_path)

        if self._tracker.is_finished:
            info_io(
                f"{self._operation_name.capitalize()} for {self._task} is already in directory:\n{str(self._storage_path)}."
            )
            if num_subjects is not None:
                subject_ids = random.sample(self._tracker.subject_ids, k=num_subjects)
            self._verbose = orig_verbose
            return ProcessedSetReader(self._storage_path, subject_ids=subject_ids)

        info_io(f"Iterative {self._operation_name}: {self._task}", level=0)
        info_io(f"{self._operation_name.capitalize()} data for task {self._task}.")

        # Tracking info
        n_processed_subjects = len(self._tracker.subject_ids)
        n_processed_stays = len(self._tracker.stay_ids)
        n_processed_samples = self._tracker.samples

        # Parallel processing logic
        def process_subject(subject_id: str):
            """_summary_"""
            _, tracking_infos = self.transform_subject(subject_id)

            if tracking_infos:
                self.save_data([subject_id])
                return subject_id, tracking_infos

            return subject_id, None

        def init(preprocessor):
            global processor_pr
            processor_pr = preprocessor

        subject_ids, excluded_subject_ids = self._get_subject_ids(
            num_subjects=num_subjects,
            subject_ids=subject_ids,
            all_subjects=reader.subject_ids,
            processed_subjects=self._tracker.subject_ids)

        info_io(f"{self._operation_name.capitalize()} timeseries data:\n"
                f"Processed subjects: {n_processed_subjects}\n"
                f"Processed stays: {n_processed_stays}\n"
                f"Processed samples: {n_processed_samples}\n"
                f"Skipped subjects: {0}")

        # Start the run
        chunksize = max(len(subject_ids) // (cpu_count() - 1), 1)
        with Pool(cpu_count() - 1, initializer=init, initargs=(self,)) as pool:
            res = pool.imap_unordered(process_subject, subject_ids, chunksize=chunksize)

            empty_subjects = 0
            missing_subjects = 0
            while True:
                try:
                    subject_id, tracker_data = next(res)
                    if tracker_data is None:
                        empty_subjects += 1
                        # Add new samples if to meet the num subjects target
                        if num_subjects is None:
                            continue
                        debug_io(f"Missing subject is: {subject_id}")
                        try:
                            subj = excluded_subject_ids.pop()
                            res = chain(res,
                                        [pool.apply_async(process_subject, args=(subj,)).get()])
                        except IndexError:
                            missing_subjects += 1
                            debug_io(
                                f"Could not replace missing subject. Excluded subjects is: {excluded_subject_ids}"
                            )
                    else:
                        n_processed_subjects += 1
                        n_processed_stays += len(tracker_data) - 1
                        n_processed_samples += tracker_data["total"]

                    info_io(
                        f"{self._operation_name.capitalize()} timeseries data:\n"
                        f"Processed subjects: {n_processed_subjects}\n"
                        f"Processed stays: {n_processed_stays}\n"
                        f"Processed samples: {n_processed_samples}\n"
                        f"Skipped subjects: {empty_subjects}",
                        flush_block=(True and not int(os.getenv("DEBUG", 0))))
                except StopIteration as e:
                    self._tracker.is_finished = True
                    info_io(
                        f"Finalized for task {self._task} in directory:\n{str(self._storage_path)}")
                    if num_subjects is not None and missing_subjects:
                        warn_io(
                            f"The subject target was not reached, missing {missing_subjects} subjects."
                        )
                    break
        self._verbose = orig_verbose
        if original_subject_ids is not None:
            original_subject_ids = list(set(original_subject_ids) & set(self._tracker.subject_ids))
        return ProcessedSetReader(self._storage_path, subject_ids=original_subject_ids)

    def _get_subject_ids(self,
                         num_subjects: int,
                         subject_ids: list,
                         all_subjects: list,
                         processed_subjects: list = list()):
        remaining_subject_ids = list(set(all_subjects) - set(processed_subjects))
        # Select subjects to process logic
        n_processed_subjects = len(processed_subjects)
        if num_subjects is not None:
            num_subjects = max(num_subjects - n_processed_subjects, 0)
            selected_subjects_ids = random.sample(remaining_subject_ids, k=num_subjects)
            remaining_subject_ids = list(set(remaining_subject_ids) - set(selected_subjects_ids))
            random.shuffle(remaining_subject_ids)
        elif subject_ids is not None:
            unknown_subjects = set(subject_ids) - set(all_subjects)
            if unknown_subjects:
                warn_io(f"Unknown subjects: {*unknown_subjects,}")
            selected_subjects_ids = list(set(subject_ids) & set(all_subjects))
            remaining_subject_ids = list(set(remaining_subject_ids) - set(selected_subjects_ids))
        else:
            selected_subjects_ids = remaining_subject_ids
        return selected_subjects_ids, remaining_subject_ids
