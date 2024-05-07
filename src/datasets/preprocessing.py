import random
import os
import pandas as pd
from copy import deepcopy
from itertools import chain
from typing import Dict
from pathlib import Path
from utils.IO import *
from utils import dict_subset
from pathos.multiprocessing import cpu_count, Pool
from preprocessing.preprocessors import MIMICPreprocessor
from .trackers import PreprocessingTracker
from .readers import ExtractedSetReader, ProcessedSetReader
from .mimic_utils import copy_subject_info

__all__ = ["compact_processing", "iterative_processing"]


def compact_processing(dataset: dict,
                       task: str,
                       phenotypes_yaml: dict,
                       subject_ids: list = None,
                       num_subjects: int = None,
                       storage_path: Path = None,
                       source_path: Path = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """_summary_

    Args:
        timeseries (pd.DataFrame): _description_
        episodic_data (pd.DataFrame): _description_
        subject_diagnoses (pd.DataFrame): _description_
        subject_icu_history (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    tracker = PreprocessingTracker(storage_path=Path(storage_path, "progress"),
                                   num_subjects=num_subjects,
                                   subject_ids=subject_ids)

    copy_subject_info(source_path, storage_path)

    if tracker.is_finished:
        info_io(f"Compact data processing already finalized in directory:\n{str(storage_path)}")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(root_path=storage_path,
                                  subject_ids=subject_ids).read_samples(read_ids=True)

    info_io(f"Compact Preprocessing: {task}", level=0)
    preprocessor = MIMICPreprocessor(task=task,
                                     storage_path=storage_path,
                                     phenotypes_yaml=phenotypes_yaml,
                                     tracker=tracker,
                                     label_type="one-hot",
                                     verbose=True)

    subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects,
                                                        subject_ids=subject_ids,
                                                        all_subjects=dataset.keys())
    assert all([len(subject) for subject in dataset.values()])
    missing_subjects = 0
    if num_subjects is not None:
        X_subjects = dict()
        y_subjects = dict()
        while not len(X_subjects) == num_subjects:
            curr_dataset = dict_subset(dataset, subject_ids)
            X, y = preprocessor.transform(dataset=curr_dataset)
            X_subjects.update(X)
            y_subjects.update(y)
            it_missing_subjects = set(X.keys()) - set(subject_ids)
            subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects -
                                                                len(X_subjects),
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
        (X_subjects, y_subjects) = preprocessor.transform(dataset=dataset)
    if storage_path is not None:
        preprocessor.save_data()
        info_io(f"Finalized data preprocessing for {task} in directory:\n{str(storage_path)}")
    else:
        info_io(f"Finalized data preprocessing for {task}.")
    # TODO! this doesn't work, reimagine
    # if missing_subjects:
    #     warn_io(f"The subject target was not reached, missing {missing_subjects} subjects.")
    tracker.is_finished = True
    return {"X": X_subjects, "y": y_subjects}


def iterative_processing(reader: ExtractedSetReader,
                         task: str,
                         phenotypes_yaml: dict,
                         subject_ids: list = None,
                         num_subjects: int = None,
                         storage_path: Path = None) -> ProcessedSetReader:
    """_summary_

    Args:
        timeseries (pd.DataFrame): _description_
        episodic_data (pd.DataFrame): _description_
        subject_diagnoses (pd.DataFrame): _description_
        subject_icu_history (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    original_subject_ids = deepcopy(subject_ids)

    tracker = PreprocessingTracker(storage_path=Path(storage_path, "progress"),
                                   num_subjects=num_subjects,
                                   subject_ids=subject_ids)
    copy_subject_info(reader.root_path, storage_path)

    preprocessor = MIMICPreprocessor(task=task,
                                     reader=reader,
                                     storage_path=storage_path,
                                     tracker=tracker,
                                     phenotypes_yaml=phenotypes_yaml,
                                     label_type="one-hot")

    if tracker.is_finished:
        info_io(f"Data preprocessing for {task} is already in directory:\n{str(storage_path)}.")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(storage_path, subject_ids=subject_ids)

    info_io(f"Iterative Preprocessing: {task}", level=0)
    info_io(f"Preprocessing data for task {task}.")

    # Tracking info
    n_processed_subjects = len(tracker.subject_ids)
    n_processed_stays = len(tracker.stay_ids)
    n_processed_samples = tracker.samples

    # Parallel processing logic
    def process_subject(subject_id: str):
        """_summary_
        """
        _, tracking_infos = preprocessor_pr.transform_subject(subject_id)

        if tracking_infos:
            preprocessor_pr.save_data([subject_id])
            return subject_id, tracking_infos

        return subject_id, None

    def init(preprocessor: MIMICPreprocessor):
        global preprocessor_pr
        preprocessor_pr = preprocessor

    subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects,
                                                        subject_ids=subject_ids,
                                                        all_subjects=preprocessor.subjects,
                                                        processed_subjects=tracker.subject_ids)

    info_io(f"Processing timeseries data:\n"
            f"Processed subjects: {n_processed_subjects}\n"
            f"Processed stays: {n_processed_stays}\n"
            f"Processed samples: {n_processed_samples}\n"
            f"Skipped subjects: {0}")

    # Start the run
    with Pool(cpu_count() - 1, initializer=init, initargs=(preprocessor,)) as pool:
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
                        res = chain(res, [pool.apply_async(process_subject, args=(subj,)).get()])
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
                    flush_block=(True and not int(os.getenv("DEBUG"))))
            except StopIteration as e:
                tracker.is_finished = True
                info_io(f"Finalized for task {task} in directory:\n{str(storage_path)}")
                if num_subjects is not None and missing_subjects:
                    warn_io(
                        f"The subject target was not reached, missing {missing_subjects} subjects.")
                break
    if original_subject_ids is not None:
        original_subject_ids = list(set(original_subject_ids) & set(tracker.subject_ids))
    return ProcessedSetReader(storage_path, subject_ids=original_subject_ids)


def get_subject_ids(num_subjects: int,
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
