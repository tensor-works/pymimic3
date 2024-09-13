import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import lru_cache
from copy import deepcopy
from metrics import CustomBins, LogBins
from utils.arrays import _transform_array
from typing import List, Tuple, Union
from datasets.trackers import PreprocessingTracker
from pathos.multiprocessing import Pool, cpu_count

__all__ = ["read_timeseries", "subjects_for_samples", "make_prediction_vector"]

def read_timeseries(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    row_only=False,
    bining="none",
    one_hot=False,
    dtype=np.ndarray,
    preserve_dtype: bool = True
) -> Tuple[List[Union[np.ndarray, pd.DataFrame]], \
           List[Union[np.ndarray, pd.DataFrame]], \
           List[Union[np.ndarray, pd.DataFrame]]]:
    """
    Reads sample and label frames time step by time step and returns them as lists.

    Parameters
    ----------
    X_df : pd.DataFrame
        The sample dataframe or array containing the features.
    y_df : pd.DataFrame
        The target dataframe or array containing the labels.
    row_only : bool, optional
        If True, return only the current sample row. If False, return the entire preceding data frame. Default is False.
    bining : str, optional
        The binning mode to apply to the labels. Options are 'none', 'custom', or 'log'. Default is 'none'.
    one_hot : bool, optional
        If True, apply one-hot encoding to categorical labels. Default is False.
    dtype : type, optional
        The type of the data contained in the returned list. Typically `np.ndarray` or `pd.DataFrame`. Default is `np.ndarray`.
    preserve_dtype : bool, optional
        If True, preserve the original data types in the target primitives. Default is True.

    Returns
    -------
    Xs : List[Union[np.ndarray, pd.DataFrame]]
        List of feature dataframes or arrays for each time step.
    ys : List[Union[np.ndarray, pd.DataFrame]]
        List of target arrays or dataframes for each time step, transformed based on binning or one-hot encoding.
    ts : List[Union[np.ndarray, pd.DataFrame]]
        List of timestamp arrays or indices corresponding to each time step.

    Examples
    --------
    >>> Xs, ys, ts = read_timeseries(X_df, y_df, row_only=True, bining="log", one_hot=True)
    >>> len(Xs), len(ys), len(ts)
    (10, 10, 10)
    """
    if bining == "log":
        y_df = y_df.apply(lambda x: LogBins.get_bin_log(x, one_hot=one_hot))
        if not isinstance(y_df, pd.DataFrame):
            y_df = y_df.to_frame()
    elif bining == "custom":
        y_df = y_df.apply(lambda x: CustomBins.get_bin_custom(x, one_hot=one_hot), axis=1)
        if not isinstance(y_df, pd.DataFrame):
            y_df = y_df.to_frame()

    if row_only:
        Xs = [
            X_df.loc[timestamp].values if dtype in [np.ndarray, np.array] else X_df.loc[timestamp]
            for timestamp in y_df.index
        ]
    else:
        Xs = [
            X_df.loc[:timestamp].values if dtype in [np.ndarray, np.array] else X_df.loc[:timestamp]
            for timestamp in y_df.index
        ]

    ys = _transform_array(y_df.values, preserve_dtype=preserve_dtype)
    ts = _transform_array(y_df.index.values, preserve_dtype=preserve_dtype)
    # ts = y_df.index.tolist()

    return Xs, ys, ts


@lru_cache(maxsize=256)  # People will probably use similar sizes
def subjects_for_samples(tracker: PreprocessingTracker,
                         target_size: int,
                         max_iter: int = 20,
                         deep_supervision: bool = False) -> Tuple[List[float], int]:
    """
    Selects subjects to match the target number of samples using the sample per subject counts of the 
    provided tracker.

    Parameters
    ----------
    tracker : PreprocessingTracker
        An object that tracks the subjects and their corresponding sample counts.
    target_size : int
        The target number of samples to be matched.
    max_iter : int, optional
        The maximum number of iterations to try in the multiprocessing pool. Default is 20.
    deep_supervision : bool, optional
        If True, groups subjects by both subject ID and stay ID for deeper supervision. Default is False.

    Returns
    -------
    best_subjects : List[float]
        List of selected subject IDs that best match the target sample size.
    best_size : int
        The total number of samples from the selected subjects.

    Notes
    -----
    - The function leverages multiprocessing to explore different random selections of subjects.

    Examples
    --------
    >>> tracker = PreprocessingTracker(subjects=subject_data)
    >>> selected_subjects, total_samples = subjects_for_samples(tracker, target_size=1000, max_iter=10, deep_supervision=True)
    >>> print(selected_subjects, total_samples)
    [1234, 2345, 3456, 4567], 1000
    """
    assert tracker.subject_ids
    # Init tracking vars
    best_diff = float('inf')
    iteration = 0

    # Init subject counts
    if deep_supervision:
        index_tuples = [(outer_key, inner_key, value)
                        for outer_key, inner_dict in deepcopy(dict(tracker.subjects)).items()
                        if not outer_key == "total" for inner_key, value in inner_dict.items()]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['subject_id', 'stay_id', 'values'])
        df = pd.DataFrame(index=index)
        df['value'] = df.index.get_level_values(2)
        df = df.droplevel(2).sort_index()
        df = df.drop(df.index[df.index.get_level_values('stay_id') == 'total'])
        # Now, group by the first index level and count entries
        subject_df = df.groupby(level='subject_id').size()
    else:
        subject_df = pd.DataFrame(deepcopy(dict(tracker.subjects))).T["total"]
        subject_df = subject_df.drop("total")

    def compute_samples(random_state):
        np.random.seed(random_state)
        current_size = 0
        remaining_subjects = subjects_df_pr
        subjects = []

        while current_size < target_size_pr and len(remaining_subjects):
            remaining_subjects = remaining_subjects[remaining_subjects <= target_size_pr -
                                                    current_size]
            if remaining_subjects.empty:
                break
            next_subject = np.random.choice(remaining_subjects.index)
            with lock_pr:
                subject_samples = remaining_subjects.loc[next_subject]

            current_size += subject_samples
            subjects.append(next_subject)
            remaining_subjects = remaining_subjects.drop(next_subject)

        diff = abs(target_size_pr - current_size)
        return diff, current_size, subjects

    # MP global vares
    def init(subject_df: pd.DataFrame, target_size: int, tracker: PreprocessingTracker,
             lock: mp.Lock):
        global subjects_df_pr, target_size_pr, tracker_pr, lock_pr
        subjects_df_pr = subject_df
        target_size_pr = target_size
        tracker_pr = tracker
        lock_pr = lock

    # Mp lock
    lock = mp.Lock()
    # Mp count
    n_cpus = cpu_count() - 1
    # Mp Pool
    with Pool(n_cpus, initializer=init, initargs=(subject_df, target_size, tracker, lock)) as pool:
        # Try max_iter times and fetch best result
        res = pool.imap_unordered(compute_samples,
                                  range(max_iter),
                                  chunksize=int(np.ceil(max_iter / n_cpus)))

        for diff, current_size, subjects in res:
            iteration += 1
            # Fetch best result
            if diff < best_diff:
                best_subjects, best_size, best_diff = subjects, current_size, diff

            # Break if no diff
            if best_diff == 0 or iteration >= max_iter:
                break
        pool.close()
        pool.join()

    # Always get smallest best, so if no best found target size is too small
    if not best_subjects:
        return subject_df.min(), subject_df.argmin()

    return best_subjects, best_size


def make_prediction_vector(model, generator, batches=20, bin_averages=None):
    """ Deprecated
    """
    # Deprecate
    Xs = list()
    ys = list()

    for _ in range(batches):
        X, y = generator.next()
        Xs.append(X)
        ys.append(y)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate([model.predict(X, verbose=0) for X in Xs])

    if bin_averages:
        y_pred = np.array([
            bin_averages[int(label)]
            if label < len(bin_averages) else bin_averages[len(bin_averages) - 1]
            for label in np.argmax(y_pred, axis=1)
        ]).reshape(1, -1)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        y_true = np.array([
            bin_averages[int(label)]
            if label < len(bin_averages) else bin_averages[len(bin_averages) - 1]
            for label in y_true
        ]).reshape(1, -1)

    return y_pred, y_true
