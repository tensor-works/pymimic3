import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, List
from pathlib import Path
from itertools import chain
from pathos.multiprocessing import cpu_count, Pool
from pathos.helpers import mp
from utils.IO import *
from utils import dict_subset
from preprocessing.feature_engines import MIMICFeatureEngine
from .trackers import PreprocessingTracker
from .readers import ProcessedSetReader
from .mimic_utils import copy_subject_info

__all__ = ["iterative_fengineering", "compact_fengineering"]


def compact_fengineering(X_subjects: Dict[str, Dict[str, pd.DataFrame]],
                         y_subjects: Dict[str, Dict[str, pd.DataFrame]],
                         task: str,
                         storage_path=None,
                         source_path=None,
                         subject_ids=None,
                         num_subjects=None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """_summary_

    Args:
        X_subjects (_type_): _description_
        y_subjects (_type_): _description_

    Returns:
        _type_: _description_
    """
    tracker = PreprocessingTracker(num_subjects=num_subjects,
                                   subject_ids=subject_ids,
                                   storage_path=Path(storage_path, "progress"))
    copy_subject_info(source_path, storage_path)

    if tracker.is_finished:
        info_io(f"Compact feature engineering already finalized in directory:\n{str(storage_path)}")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(root_path=storage_path,
                                  subject_ids=subject_ids).read_samples(read_ids=True)

    info_io(f"Compact Feature Engineering: {task}", level=0)
    engine = MIMICFeatureEngine(config_dict=Path(os.getenv("CONFIG"), "engineering_config.json"),
                                storage_path=storage_path,
                                task=task,
                                tracker=tracker,
                                verbose=True)

    subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects,
                                                        subject_ids=subject_ids,
                                                        all_subjects=X_subjects.keys())

    X_subjects = dict_subset(X_subjects, subject_ids)
    y_subjects = dict_subset(y_subjects, subject_ids)


    X_processed, \
    y_processed, \
    _ = engine.transform(X_subjects, y_subjects) # Omitting timestamps

    if storage_path:
        engine.save_data()
        info_io(f"Finalized feature engineering for {task} in directory:\n{str(storage_path)}")
    else:
        info_io(f"Finalized feature engineering for {task}.")
    tracker.is_finished = True
    return {"X": X_processed, "y": y_processed}


def iterative_fengineering(reader: ProcessedSetReader,
                           task: str,
                           storage_path: Path,
                           subject_ids: List[int] = None,
                           num_subjects: int = None):
    """_summary_
    """
    original_subject_ids = deepcopy(subject_ids)
    tracker = PreprocessingTracker(storage_path=Path(storage_path, "progress"),
                                   num_subjects=num_subjects,
                                   subject_ids=subject_ids)

    copy_subject_info(reader.root_path, storage_path)

    engine = MIMICFeatureEngine(reader=reader,
                                config_dict=Path(os.getenv("CONFIG"), "engineering_config.json"),
                                storage_path=storage_path,
                                task=task,
                                tracker=tracker)

    if tracker.is_finished:
        info_io(f"Data engineering for {task} is already in directory:\n{str(storage_path)}.")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(storage_path, subject_ids=subject_ids)

    info_io(f"Iterative Feature Engineering: {task}", level=0)
    info_io(f"Engineering data and saving at:\n{storage_path}.")

    # Tracking info
    n_engineered_subjects = len(tracker.subject_ids)
    n_engineered_stays = len(tracker.stay_ids)
    n_engineered_samples = tracker.samples

    def engineer_subject(subject_id: str):
        """"""

        _, tracking_infos = engine_pr.transform_subject(subject_id)

        if tracking_infos is not None:
            engine_pr.save_data([subject_id])
            return subject_id, tracking_infos

        return subject_id, None

    def init(engine: MIMICFeatureEngine):
        global engine_pr
        engine_pr = engine

    # Select subjects to process logic
    subject_ids, excluded_subject_ids = get_subject_ids(num_subjects=num_subjects,
                                                        subject_ids=subject_ids,
                                                        all_subjects=engine.subjects,
                                                        engineered_subjects=tracker.subject_ids)

    info_io(f"Engineering processed data:\n"
            f"Engineered subjects: {n_engineered_subjects}\n"
            f"Engineered stays: {n_engineered_stays}\n"
            f"Engineered samples: {n_engineered_samples}")
    with Pool(cpu_count() - 1, initializer=init, initargs=(engine,)) as pool:
        chunksize = min(1000, int(np.ceil(len(subject_ids) / (cpu_count() - 1))) + 1)
        res = pool.imap_unordered(engineer_subject, subject_ids, chunksize=chunksize)

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
                    try:
                        subj = excluded_subject_ids.pop()
                        res = chain(res, [pool.apply_async(engineer_subject, args=(subj,)).get()])
                    except IndexError:
                        debug_io(f"Missing subject is: {subject_id}")
                        missing_subjects += 1
                else:
                    n_engineered_subjects += 1
                    n_engineered_stays += len(tracker_data) - 1
                    n_engineered_samples += tracker_data["total"]

                info_io(
                    f"Engineering processed data:\n"
                    f"Engineered subjects: {n_engineered_subjects}\n"
                    f"Engineered stays: {n_engineered_stays}\n"
                    f"Engineered samples: {n_engineered_samples}\n"
                    f"Skipped subjects: {empty_subjects}",
                    flush_block=True)
            except StopIteration:
                tracker.is_finished = True
                info_io(
                    f"Finalized feature engineering for {task} in directory:\n{str(storage_path)}")
                if num_subjects is not None and missing_subjects:
                    info_io(
                        f"The subject target was not reached, missing {missing_subjects} subjects.")
                break
    if original_subject_ids is not None:
        original_subject_ids = list(set(original_subject_ids) & set(tracker.subject_ids))
    return ProcessedSetReader(storage_path, original_subject_ids)


def get_subject_ids(num_subjects: int,
                    subject_ids: list,
                    all_subjects: list,
                    engineered_subjects: list = list()):
    """_summary_

    Args:
        num_subjects (_type_): _description_
        subject_ids (_type_): _description_
        all_subjects (_type_): _description_

    Returns:
        _type_: _description_
    """
    remaining_subject_ids = list(set(all_subjects) - set(engineered_subjects))
    n_engineered_subjects = len(engineered_subjects)
    if num_subjects is not None:
        num_subjects = max(num_subjects - n_engineered_subjects, 0)
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
