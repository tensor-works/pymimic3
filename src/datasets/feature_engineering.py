import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
from pathos.multiprocessing import cpu_count, Pool
from pathos.helpers import mp
from utils.IO import *
from utils import dict_subset
from preprocessing.feature_engines import MIMICFeatureEngine
from .trackers import PreprocessingTracker
from .readers import ProcessedSetReader

__all__ = ["iterative_fengineering", "compact_fengineering"]


def compact_fengineering(X_subjects: pd.DataFrame,
                         y_subjects: pd.DataFrame,
                         task: str,
                         storage_path=None,
                         subject_ids=None,
                         num_subjects=None):
    """_summary_

    Args:
        X_subjects (_type_): _description_
        y_subjects (_type_): _description_

    Returns:
        _type_: _description_
    """
    tracker = PreprocessingTracker(num_subjects=num_subjects,
                                   subject_ids=subject_ids,
                                   storage_path=storage_path)

    if tracker.finished:
        info_io(f"Compact feature engineering already finalized in directory:\n{str(storage_path)}")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(root_path=storage_path,
                                  subject_folders=subject_ids).read_samples(read_ids=True)

    info_io("Compact Feature Engineering", level=0)
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
    tracker.finished = True
    return {"X": X_processed, "y": y_processed}


def iterative_fengineering(reader,
                           task,
                           storage_path,
                           subject_ids=None,
                           num_subjects=None,
                           num_samples=None):
    """_summary_
    """
    tracker = PreprocessingTracker(storage_path=storage_path,
                                   num_subjects=num_subjects,
                                   subject_ids=subject_ids)

    engine = MIMICFeatureEngine(reader=reader,
                                config_dict=Path(os.getenv("CONFIG"), "engineering_config.json"),
                                storage_path=storage_path,
                                task=task,
                                tracker=tracker)

    if tracker.finished:
        info_io(f"Data engineering for {task} is already in directory:\n{str(storage_path)}.")
        if num_subjects is not None:
            subject_ids = random.sample(tracker.subject_ids, k=num_subjects)
        return ProcessedSetReader(storage_path, subject_folders=subject_ids)

    info_io("Iterative Feature Engineering", level=0)
    info_io(f"Engineering data and saving at:\n{storage_path}.")

    # Tracking info
    n_engineered_subjects = len(tracker.subject_ids)
    n_engineered_stays = len(tracker.stay_ids)
    n_engineered_samples = tracker.samples

    # engine_pr = engine

    def engineer_subject(subject_id: str):
        """"""

        _, tracking_infos = engine_pr.transform_subject(subject_id)

        if tracking_infos is not None:
            engine_pr.save_data([subject_id])
            return subject_id, tracking_infos

        return subject_id, None

    # for id in remaining_subject_ids:
    #     engineer_subject(id)

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
                    except IndexError:
                        debug_io(f"Missing subject is: {subject_id}")
                        missing_subjects += 1
                        continue
                    res = chain(res, [pool.apply_async(engineer_subject, args=(subj,)).get()])
                    continue
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
                tracker.finished = True
                info_io(
                    f"Finalized feature engineering for {task} in directory:\n{str(storage_path)}")
                if num_subjects is not None and missing_subjects:
                    info_io(
                        f"The subject target was not reached, missing {missing_subjects} subjects.")
                break

    return ProcessedSetReader(storage_path)


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
        unknown_subjects = (set(remaining_subject_ids) - set(subject_ids)
                           ) & set(set(remaining_subject_ids) - set(n_engineered_subjects))
        if unknown_subjects:
            warn_io(f"Unknown subjects: {*unknown_subjects,}")
        selected_subjects_ids = subject_ids
    else:
        selected_subjects_ids = remaining_subject_ids
    return selected_subjects_ids, remaining_subject_ids
