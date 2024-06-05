import random
import os
import pandas as pd
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Tuple
from utils import dict_subset
from pathlib import Path
from abc import ABC, abstractmethod
from pathos.multiprocessing import cpu_count, Pool
from datasets.readers import ExtractedSetReader, ProcessedSetReader
from datasets.mimic_utils import copy_subject_info
from utils.IO import *


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

    def transform_dataset(self,
                          dataset: dict,
                          subject_ids: list = None,
                          num_subjects: int = None,
                          source_path: Path = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        _summary_

        Parameters
        ----------
        dataset : dict
            The dataset to process.
        subject_ids : list, optional
            List of subject IDs. Defaults to None.
        num_subjects : int, optional
            Number of subjects to process. Defaults to None.
        source_path : Path, optional
            Source path of the data. Defaults to None.

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Processed data.
        """
        copy_subject_info(source_path, self._storage_path)

        if self._tracker.is_finished:
            info_io(
                f"Compact data processing already finalized in directory:\n{str(self._storage_path)}"
            )
            if num_subjects is not None:
                subject_ids = random.sample(self._tracker.subject_ids, k=num_subjects)
            return ProcessedSetReader(root_path=self._storage_path,
                                      subject_ids=subject_ids).read_samples(read_ids=True)

        info_io(f"Compact Preprocessing: {self._task}", level=0)

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
                f"Finalized data preprocessing for {self._task} in directory:\n{str(self._storage_path)}"
            )
        else:
            info_io(f"Finalized data preprocessing for {self._task}.")
        self._tracker.is_finished = True
        return {"X": X_subjects, "y": y_subjects}

    def transform_reader(self,
                         reader: ExtractedSetReader,
                         subject_ids: list = None,
                         num_subjects: int = None) -> ProcessedSetReader:
        """
        _summary_

        Parameters
        ----------
        reader : ExtractedSetReader
            Reader for the extracted set.
        subject_ids : list, optional
            List of subject IDs. Defaults to None.
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
                f"Data preprocessing for {self._task} is already in directory:\n{str(self._storage_path)}."
            )
            if num_subjects is not None:
                subject_ids = random.sample(self._tracker.subject_ids, k=num_subjects)
            self._verbose = orig_verbose
            return ProcessedSetReader(self._storage_path, subject_ids=subject_ids)

        info_io(f"Iterative Preprocessing: {self._task}", level=0)
        info_io(f"Preprocessing data for task {self._task}.")

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

        for ids in subject_ids:
            process_subject(ids)

        info_io(f"Processing timeseries data:\n"
                f"Processed subjects: {n_processed_subjects}\n"
                f"Processed stays: {n_processed_stays}\n"
                f"Processed samples: {n_processed_samples}\n"
                f"Skipped subjects: {0}")

        # Start the run
        with Pool(cpu_count() - 1, initializer=init, initargs=(self,)) as pool:
            res = pool.imap_unordered(process_subject, subject_ids, chunksize=500)

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
                        f"Processing timeseries data:\n"
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
