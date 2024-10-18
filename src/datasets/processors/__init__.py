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
import os
import pandas as pd
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Tuple, Union
from settings import *
from utils.jsons import dict_subset
from pathlib import Path
from abc import ABC, abstractmethod
from pathos.multiprocessing import cpu_count, Pool
from datasets.readers import ExtractedSetReader, ProcessedSetReader
from datasets.trackers import PreprocessingTracker
from datasets.writers import DataSetWriter
from utils.IO import *
from pathos.helpers import mp
from multiprocess import Manager


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
        self._writer: DataSetWriter = ...
        self._save_file_type: str = ...
        self._operation_adjective: str = ...
        self._lock: Manager.Lock = ...
        self._X: dict = ...
        self._y: dict = ...

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
    def _transform(self, *args, **kwargs):
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

    def _init_tracking_variables(self, subject_ids: list = None):
        if subject_ids is None:
            # Tracking variables
            self._n_subjects = len(self._tracker.subject_ids)
            self._n_stays = len(self._tracker.stay_ids)
            self._n_samples = self._tracker.samples
            self._n_skip = 0
        else:
            proc_subjects = set(subject_ids) & set(self._tracker.subject_ids)
            self._n_subjects = len(proc_subjects)
            self._n_stays = sum([
                1 for subject_id in proc_subjects for stay_id in self._tracker.subjects[subject_id]
                if stay_id != "total"
            ])
            self._n_samples = sum(
                [self._tracker.subjects[subject_id]["total"] for subject_id in proc_subjects])
            self._n_skip = 0

    def save_data(self, subject_ids: list = None) -> None:
        """
        Save the discretized data to the storage path.

        If no subjects are specified, all the discretized data will be saved.

        Parameters
        ----------
        subjects : list, optional
            A list of subject IDs to save data for. If None, all data is saved. Default is None.
        """
        if self._writer is None:
            info_io("No storage path provided. Data will not be saved.", verbose=self._verbose)
            return

        def save_dict(data: dict, key: str, subject_ids: list = None):
            if subject_ids is None:
                data_save = data
                subject_ids = list(data.keys())
            else:
                data_save = dict_subset(data, subject_ids)
            self._writer.write_bysubject({key: data_save}, file_type=self._save_file_type)
            for subject in subject_ids:
                del data[subject]

        with self._lock:
            save_dict(self._X, "X", subject_ids)
            if hasattr(self, "_deep_supervision") and self._deep_supervision:
                save_dict(self._M, "M", subject_ids)
                save_dict(self._y, "yds", subject_ids)
            else:
                save_dict(self._y, "y", subject_ids)
            if hasattr(self, "_t"):
                save_dict(self._t, "t", subject_ids)

        return

    def transform_subject(self, subject_id: int, return_tracking=False):
        """
        Transform the data for a specific subject.

        This method reads the data for a specific subject, processes it, and returns
        the engineered features along with tracking information.

        Parameters
        ----------
        subject_id : int
            The ID of the subject to transform data for.

        Returns
        -------
        tuple
            A tuple containing the engineered features and tracking information.
        """
        if isinstance(self._source_reader, ProcessedSetReader):
            subject_data = self._source_reader.read_samples([subject_id],
                                                            read_ids=True,
                                                            data_type=pd.DataFrame)
        elif isinstance(self._source_reader, ExtractedSetReader):
            subject_data = self._source_reader.read_subjects([subject_id], read_ids=True)

        proc_data, tracking_info = self._transform(subject_data, return_tracking=True)
        if return_tracking:
            return proc_data, tracking_info
        return proc_data

    def _update_tracking(self, subject_id: int, tracking_info: dict, overwrite: bool = True):
        # Common logic for updating the tracking info and removing empty subjects
        if subject_id in tracking_info:
            if tracking_info[subject_id]:
                self._n_subjects += 1
                if self._tracker is not None:
                    with self._lock:
                        if not subject_id in self._tracker.subjects or overwrite:
                            self._tracker.subjects.update({subject_id: tracking_info[subject_id]})
                            self._tracker._to_dict()
            else:
                self._n_skip += 1
                del tracking_info[subject_id]
                del self._y[subject_id]
                del self._X[subject_id]
                if hasattr(self, "_deep_supervision") and self._deep_supervision:
                    del self._M[subject_id]

        return tracking_info

    def transform_dataset(self,
                          dataset: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
                          storage_path=None,
                          subject_ids=None,
                          num_subjects=None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Transforms the dataset provided by the previous processing stage directly.

        This method processes the provided data, transforming it for each subject, and then saving 
        the processed data if a storage path is provided. It handles subject selection and ensures 
        the processed data is correctly stored.

        Parameters
        ----------
        dataset : Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
            Processed dataset.
        subject_ids : list, optional
            List of subject IDs to process. Defaults to None.
        num_subjects : int, optional
            Number of subjects to process. Defaults to None.

        Returns
        -------
        ProcessedSetReader
            Reader for the processed set.
        """
        # TODO! This also needs to work with no storage path provided
        X_subjects, y_subjects = list(dataset.values())
        orig_subject_ids = deepcopy(subject_ids)
        self._init_tracking_variables(subject_ids)
        if storage_path is not None:
            self._storage_path = Path(storage_path)
            self._tracker = PreprocessingTracker(storage_path=Path(storage_path, "progress"))
            self._tracker.set_subject_ids(subject_ids)
            self._tracker.set_num_subjects(num_subjects)
        elif self._storage_path is not None and (num_subjects is not None or
                                                 subject_ids is not None):
            self._tracker.set_subject_ids(subject_ids)
            self._tracker.set_num_subjects(num_subjects)

        if self._tracker.is_finished:
            info_io(
                f"Compact {self._operation_name}  already finalized in directory:\n{str(self._storage_path)}",
                verbose=self._verbose)
            if num_subjects is not None:
                subject_ids = random.sample(self._tracker.subject_ids, k=num_subjects)
            return ProcessedSetReader(root_path=self._storage_path,
                                      subject_ids=subject_ids).read_samples(read_ids=True)

        info_io(f"Compact {self._operation_name}: {self._task}", level=0, verbose=self._verbose)

        subject_ids, exclud_subj, unkonwn_subj = self._get_subject_ids(
            num_subjects=num_subjects,
            subject_ids=subject_ids,
            processed_subjects=self._tracker.subject_ids
            if not self._tracker.force_rerun else list(),
            all_subjects=X_subjects.keys())

        if not subject_ids:
            self._tracker.is_finished = True
            info_io(f"Finalized for task {self._task} in directory:\n{str(self._storage_path)}",
                    verbose=self._verbose)
            if num_subjects and not self._n_subjects == num_subjects:
                warn_io(
                    f"The subject target was not reached, missing {self._n_subjects - num_subjects} subjects.",
                    verbose=self._verbose)
            if orig_subject_ids is not None:
                orig_subject_ids = list(set(orig_subject_ids) & set(self._tracker.subject_ids))
            return ProcessedSetReader(self._storage_path,
                                      subject_ids=orig_subject_ids).read_samples(read_ids=True)

        self._n_skip = len(unkonwn_subj)

        X_subjects = dict_subset(X_subjects, subject_ids)
        y_subjects = dict_subset(y_subjects, subject_ids)

        self._transform({"X": X_subjects, "y": y_subjects})  # Omitting timestamps

        if storage_path or self._storage_path:
            self.save_data()
            info_io(
                f"{self._operation_name.capitalize()} engineering for {self._task} in directory:\n{str(self._storage_path)}",
                verbose=self._verbose)
        else:
            info_io(f"Finalized {self._operation_name} for {self._task}.", verbose=self._verbose)
        self._tracker.is_finished = True
        # TODO! inefficient
        if orig_subject_ids is not None:
            orig_subject_ids = list(set(orig_subject_ids) & set(self._tracker.subject_ids))
        return ProcessedSetReader(root_path=self._storage_path,
                                  subject_ids=orig_subject_ids).read_samples(read_ids=True)

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
        self._init_tracking_variables(subject_ids)
        if subject_ids is not None or num_subjects is not None:
            # Reinit the tracker to check if reprocessing is necessary
            self._tracker.set_subject_ids(subject_ids)
            self._tracker.set_num_subjects(num_subjects)

        if self._tracker.is_finished:
            info_io(
                f"{self._operation_name.capitalize()} for {self._task} is already in directory:\n{str(self._storage_path)}.",
                verbose=orig_verbose)
            if num_subjects is not None:
                subject_ids = random.sample(self._tracker.subject_ids, k=num_subjects)
            self._verbose = orig_verbose
            return ProcessedSetReader(self._storage_path, subject_ids=subject_ids)

        info_io(f"Iterative {self._operation_name}: {self._task}", level=0, verbose=orig_verbose)
        info_io(f"{self._operation_name.capitalize()} data for task {self._task}.",
                verbose=orig_verbose)

        # Parallel processing logic
        def process_subject(subject_id: str):
            """_summary_"""
            _, tracking_infos = self.transform_subject(subject_id, return_tracking=True)

            if tracking_infos:
                self.save_data([subject_id])
                # Add total sample count
                tracking_infos["total"] = sum([
                    data for stay_id, data in tracking_infos[subject_id].items()
                    if stay_id != "total"
                ])
                return subject_id, tracking_infos
            # Return empty tracking info
            return subject_id, tracking_infos

        def init(preprocessor):
            global processor_pr
            processor_pr = preprocessor

        subject_ids, exclud_subj, unknown_subj = self._get_subject_ids(
            num_subjects=num_subjects,
            subject_ids=subject_ids,
            all_subjects=reader.subject_ids,
            processed_subjects=self._tracker.subject_ids
            if not self._tracker.force_rerun else list())

        # for subject_id in subject_ids:
        #     process_subject(subject_id)

        if not subject_ids:
            self._verbose = orig_verbose
            self._tracker.is_finished = True
            info_io(f"Finalized for task {self._task} in directory:\n{str(self._storage_path)}",
                    verbose=orig_verbose)
            if num_subjects and not self._n_subjects == num_subjects:
                warn_io(
                    f"The subject target was not reached, missing {self._n_subjects - num_subjects} subjects.",
                    verbose=orig_verbose)
            if original_subject_ids is not None:
                original_subject_ids = list(
                    set(original_subject_ids) & set(self._tracker.subject_ids))
            return ProcessedSetReader(self._storage_path, subject_ids=original_subject_ids)

        missing_subjects = len(unknown_subj)
        info_io(
            f"{self._operation_name.capitalize()} timeseries data:\n"
            f"{self._operation_adjective.capitalize()} subjects: {self._n_subjects}\n"
            f"{self._operation_adjective.capitalize()} stays: {self._n_stays}\n"
            f"{self._operation_adjective.capitalize()} samples: {self._n_samples}\n"
            f"Skipped subjects: {len(unknown_subj)}",
            verbose=orig_verbose)

        # Start the run
        chunksize = max(len(subject_ids) // (cpu_count() - 1), 1)
        with Pool(cpu_count() - 1, initializer=init, initargs=(self,)) as pool:
            res = pool.imap_unordered(process_subject, subject_ids, chunksize=chunksize)

            while True:
                try:
                    subject_id, tracker_info = next(res)
                    if not tracker_info:
                        # Subject could not be transformed
                        # Add new samples if to meet the num subjects target
                        if num_subjects is None:
                            # No subject target was set lets move on
                            missing_subjects += 1
                            continue
                        debug_io(f"Missing subject is: {subject_id}", verbose=orig_verbose)
                        try:

                            # Try to replace the missing subject to meet target
                            subj = exclud_subj.pop()
                            res = chain(res,
                                        [pool.apply_async(process_subject, args=(subj,)).get()])
                        except IndexError:
                            missing_subjects += 1
                            debug_io(
                                f"Could not replace missing subject. Excluded subjects is: {exclud_subj}",
                                verbose=orig_verbose)
                    elif subject_id in tracker_info:
                        self._n_subjects += 1
                        self._n_stays += len(tracker_info[subject_id])
                        self._n_samples += tracker_info["total"]

                    info_io(
                        f"{self._operation_name.capitalize()} timeseries data:\n"
                        f"{self._operation_adjective.capitalize()} subjects: {self._n_subjects}\n"
                        f"{self._operation_adjective.capitalize()} stays: {self._n_stays}\n"
                        f"{self._operation_adjective.capitalize()} samples: {self._n_samples}\n"
                        f"Skipped subjects: {missing_subjects}",
                        flush_block=(True and not int(os.getenv("DEBUG", 0))),
                        verbose=orig_verbose)
                except StopIteration as e:
                    self._tracker.is_finished = True
                    info_io(
                        f"Finalized for task {self._task} in directory:\n{str(self._storage_path)}",
                        verbose=orig_verbose)
                    if num_subjects is not None and missing_subjects:
                        warn_io(
                            f"The subject target was not reached, missing {missing_subjects} subjects.",
                            verbose=orig_verbose)
                    pool.close()
                    pool.join()
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
            if num_subjects > len(remaining_subject_ids):
                warn_io(
                    f"Number of requested subjects ({num_subjects + n_processed_subjects}) exceeds available subjects ({len(all_subjects)})"
                )
                num_subjects = len(remaining_subject_ids)
            selected_subjects_ids = random.sample(remaining_subject_ids, k=num_subjects)
            remaining_subject_ids = list(set(remaining_subject_ids) - set(selected_subjects_ids))
            random.shuffle(remaining_subject_ids)
            unknown_subjects = list()
        elif subject_ids is not None:
            unknown_subjects = set(subject_ids) - set(all_subjects)
            if unknown_subjects:
                warn_io(f"Unknown subjects: {*unknown_subjects,}")
            selected_subjects_ids = list((set(subject_ids) & set(remaining_subject_ids)))
            remaining_subject_ids = list(set(remaining_subject_ids) - set(selected_subjects_ids))
        else:
            selected_subjects_ids = remaining_subject_ids
            unknown_subjects = list()
        return selected_subjects_ids, remaining_subject_ids, unknown_subjects
