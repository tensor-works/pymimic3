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
from preprocessing.discretizers import MIMICDiscretizer
from .trackers import PreprocessingTracker
from .readers import ExtractedSetReader, ProcessedSetReader
from .mimic_utils import copy_subject_info

__all__ = ["compact_discretization", "iterative_discretization"]


def compact_discretization(X_subject: Dict[str, Dict[str, pd.DataFrame]],
                           y_subject: Dict[str, Dict[str, pd.DataFrame]],
                           task: str,
                           subject_ids: list = None,
                           num_subjects: int = None,
                           time_step_size: float = 1.0,
                           impute_strategy: str = "previous",
                           mode: str = "legacy",
                           start_at_zero: bool = True,
                           eps: float = 1e-6,
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
                                   subject_ids=subject_ids,
                                   time_step_size=time_step_size,
                                   impute_strategy=impute_strategy,
                                   start_at_zero=start_at_zero,
                                   mode=mode)

    copy_subject_info(source_path, storage_path)

    if tracker.is_finished:
        info_io(f"Compact discretization finalized in directory:\n{str(storage_path)}")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(root_path=storage_path, subject_ids=subject_ids,
                                  set_index=False).read_samples(read_ids=True)

    info_io(f"Compact Discretization: {task}", level=0)
    discretizer = MIMICDiscretizer(task=task,
                                   storage_path=storage_path,
                                   tracker=tracker,
                                   time_step_size=time_step_size,
                                   impute_strategy=impute_strategy,
                                   start_at_zero=start_at_zero,
                                   mode=mode,
                                   eps=eps,
                                   verbose=True)

    subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects,
                                                        subject_ids=subject_ids,
                                                        all_subjects=X_subject.keys())
    missing_subjects = 0
    if num_subjects is not None:
        X_discretized = dict()
        y_discretized = dict()
        while not len(X_discretized) == num_subjects:
            curr_X_subject = dict_subset(X_subject, subject_ids)
            curr_y_subject = dict_subset(y_subject, subject_ids)

            X, y = discretizer.transform(curr_X_subject, curr_y_subject)
            X_discretized.update(X)
            y_discretized.update(y)
            it_missing_subjects = set(X.keys()) - set(subject_ids)
            subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects -
                                                                len(X_discretized),
                                                                subject_ids=None,
                                                                all_subjects=excluded_subject_ids)
            if it_missing_subjects:
                missing_subjects += len(it_missing_subjects)
                debug_io(f"Missing subjects are: {*it_missing_subjects,}")
            if not subject_ids:
                break
            if len(X_discretized) == num_subjects:
                debug_io(f"Missing {len(X_discretized) - num_subjects} subjects.")
                debug_io(f"Unprocessable subjects are: {*it_missing_subjects,}")

    else:
        X_subject = dict_subset(X_subject, subject_ids)
        y_subject = dict_subset(y_subject, subject_ids)

        (X_discretized, y_discretized) = discretizer.transform(X_subject, y_subject)
    if storage_path is not None:
        discretizer.save_data()
        info_io(f"Finalized discretization for {task} in directory:\n{str(storage_path)}")
    else:
        info_io(f"Finalized discretization for {task}.")
    # TODO! this doesn't work, reimagine
    # if missing_subjects:
    #     warn_io(f"The subject target was not reached, missing {missing_subjects} subjects.")
    tracker.is_finished = True
    return {"X": X_discretized, "y": y_discretized}


def iterative_discretization(reader: ExtractedSetReader,
                             task: str,
                             subject_ids: list = None,
                             num_subjects: int = None,
                             time_step_size: float = 1.0,
                             impute_strategy: str = "previous",
                             mode: str = "legacy",
                             start_at_zero: bool = True,
                             eps: float = 1e-6,
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
                                   subject_ids=subject_ids,
                                   time_step_size=time_step_size,
                                   impute_strategy=impute_strategy,
                                   start_at_zero=start_at_zero,
                                   mode=mode)

    copy_subject_info(reader.root_path, storage_path)

    discretizer = MIMICDiscretizer(reader=reader,
                                   task=task,
                                   storage_path=storage_path,
                                   tracker=tracker,
                                   time_step_size=time_step_size,
                                   impute_strategy=impute_strategy,
                                   start_at_zero=start_at_zero,
                                   mode=mode,
                                   eps=eps,
                                   verbose=False)

    if tracker.is_finished:
        info_io(f"Data discretization for {task} is already in directory:\n{str(storage_path)}.")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(storage_path, subject_ids=subject_ids, set_index=False)

    info_io(f"Iterative Discretization: {task}", level=0)
    info_io(f"Discretizing for task {task}.")

    # Tracking info
    n_discretizer_subjects = len(tracker.subject_ids)
    n_discretizer_stays = len(tracker.stay_ids)
    n_discretizer_samples = tracker.samples

    # Parallel processing logic
    # discretizer_pr = discretizer

    def discretize_subject(subject_id: str):
        """_summary_
        """
        _, tracking_infos = discretizer_pr.transform_subject(subject_id)

        if tracking_infos:
            discretizer_pr.save_data([subject_id])
            return subject_id, tracking_infos

        return subject_id, None

    def init(discretizer: MIMICDiscretizer):
        global discretizer_pr
        discretizer_pr = discretizer

    subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects,
                                                        subject_ids=subject_ids,
                                                        all_subjects=discretizer.subjects,
                                                        discretizer_subjects=tracker.subject_ids)
    # for subject_id in subject_ids:
    #     discretize_subject(subject_id)
    info_io(f"Discretizing task data:\n"
            f"Discretize subjects: {n_discretizer_subjects}\n"
            f"Discretize stays: {n_discretizer_stays}\n"
            f"Discretize samples: {n_discretizer_samples}\n"
            f"Skipped subjects: {0}")

    # Start the run
    with Pool(cpu_count() - 1, initializer=init, initargs=(discretizer,)) as pool:
        res = pool.imap_unordered(discretize_subject, subject_ids, chunksize=500)

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
                        res = chain(res, [pool.apply_async(discretize_subject, args=(subj,)).get()])
                    except IndexError:
                        missing_subjects += 1
                        debug_io(
                            f"Could not replace missing subject. Excluded subjects is: {excluded_subject_ids}"
                        )
                else:
                    n_discretizer_subjects += 1
                    n_discretizer_stays += len(tracker_data) - 1
                    n_discretizer_samples += tracker_data["total"]

                info_io(
                    f"Discretizing timeseries data:\n"
                    f"Discretized subjects: {n_discretizer_subjects}\n"
                    f"Discretized stays: {n_discretizer_stays}\n"
                    f"Discretized samples: {n_discretizer_samples}\n"
                    f"Skipped subjects: {empty_subjects}",
                    flush_block=(True and not int(os.getenv("DEBUG", 0))))
            except StopIteration as e:
                tracker.is_finished = True
                info_io(f"Finalized for task {task} in directory:\n{str(storage_path)}")
                if num_subjects is not None and missing_subjects:
                    warn_io(
                        f"The subject target was not reached, missing {missing_subjects} subjects.")
                break

    if original_subject_ids is not None:
        original_subject_ids = list(set(original_subject_ids) & set(tracker.subject_ids))
    return ProcessedSetReader(storage_path, subject_ids=original_subject_ids, set_index=False)


def get_subject_ids(num_subjects: int,
                    subject_ids: list,
                    all_subjects: list,
                    discretizer_subjects: list = list()):
    remaining_subject_ids = list(set(all_subjects) - set(discretizer_subjects))
    # Select subjects to process logic
    n_processed_subjects = len(discretizer_subjects)
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
