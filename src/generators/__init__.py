import numpy as np
import pandas as pd
import random
import ray
import logging

from copy import deepcopy
from pathlib import Path
from utils.IO import *
from typing import List, Tuple, Union
from utils import zeropad_samples, read_timeseries
from preprocessing.scalers import AbstractScaler
from datasets.trackers import PreprocessingTracker
from datasets.readers import ProcessedSetReader
import multiprocessing as mp
from pathos.multiprocessing import Pool, cpu_count


class AbstractGenerator:

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler,
                 num_cpus: int = None,
                 batch_size: int = 8,
                 n_samples: int = None,
                 shuffle: bool = True,
                 bining: str = "none",
                 deep_supervision: bool = False,
                 target_replication: bool = False):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._target_replication = target_replication
        self._reader = reader
        self._columns = None
        self._deep_supervision = deep_supervision
        self._tracker = PreprocessingTracker(storage_path=Path(reader.root_path, "progress"))
        self._subject_ids = reader.subject_ids
        self._scaler = scaler

        # n_samples
        self._n_samples = n_samples
        if n_samples is not None:
            self._random_ids, _ = self._subjects_for_samples(self._n_samples)

        else:
            self._random_ids = deepcopy(self._reader.subject_ids)
        self._steps = self._count_batches(self._random_ids)
        random.shuffle(self._random_ids)
        self._row_only = False

        # MP setup
        if num_cpus:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_cpus=mp.cpu_count() - 1)

            ray_res = ray.cluster_resources()
            ray_cpu = int(ray_res.get("CPU", 0))
            if num_cpus is None:
                self._cpu_count = min(max(1, (ray_cpu - 1)), ray_cpu)
            else:
                self._cpu_count = min(num_cpus, ray_cpu)
        else:
            self._cpu_count = 0
        self._ray_workers = list()
        if bining not in ["none", "log", "custom"]:
            raise ValueError("Bining must be one of ['none', 'log', 'custom']")

        # State vars
        self._bining = bining
        self._counter = 0
        self._remainder_X = np.array([])
        self._remainder_y = np.array([])
        self._remainder_M = np.array([])
        self._generator = self.__generator()

    def __getitem__(self, index=None):
        if not self._ray_workers and self._cpu_count:
            self._create_workers()
            self._start_epoch()

        # Start with any remainder from the previous batch
        X, y, M = next(self._generator)  # if not deepsupervsion m is timestamps else mask
        # Fetch new data until we have at least the required batch size
        while X.shape[0] < self._batch_size:
            X_res = self._remainder_X
            y_res = self._remainder_y
            X = self._stack_batches((X, X_res)) if X_res.size else X
            if self._deep_supervision or self._target_replication:
                if self._deep_supervision:
                    m_res = self._remainder_M
                    M = self._stack_batches((M, m_res)) if m_res.size else M
                y = self._stack_batches((y, y_res)) if y_res.size else y
            else:
                y = np.concatenate((y, y_res), axis=0) if y_res.size else y
            if X.shape[0] < self._batch_size:
                self._remainder_X, \
                self._remainder_y, \
                self._remainder_M = next(self._generator)

            # If the accumulated batch is larger than required, split it
            if X.shape[0] > self._batch_size:
                self._remainder_X = X[self._batch_size:, :, :]
                self._remainder_y = y[self._batch_size:]
                X = X[:self._batch_size]
                y = y[:self._batch_size]
                if self._deep_supervision:
                    self._remainder_M = M[self._batch_size:]
                    M = M[:self._batch_size]

                break

        self._counter += 1
        if self._counter >= self._steps:
            if self._cpu_count:
                self._close()
            self._counter = 0
            self._remainder_X = np.array([])
            self._remainder_y = np.array([])
            self._remainder_M = np.array([])

        if self._deep_supervision:
            return X, y, M
        return X, y

    def _count_batches(self, subject_ids):
        # Call this only
        if self._deep_supervision:
            return max(sum([len(self._tracker.subjects[subject_id] - 1 \
                       for subject_id in subject_ids)]) \
                       // self._batch_size, 1)
        return sum([self._tracker.subjects[subject_id]["total"] for subject_id in subject_ids
                   ]) // self._batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self._steps

    def __del__(self):
        if self._cpu_count:
            self._close()

    def _create_workers(self):
        '''
        try:
            worker = RayWorkerDebug(self._reader, self._scaler, self._row_only, self._bining,
                                    self._columns, self._deep_supervision, self._target_replication)
            if self._deep_supervision:
                gen1 = worker.process_subject_deep_supervision((self._random_ids, self._batch_size))
                for _ in range(len(self._random_ids) // self._batch_size):
                    print(_)
                    next(gen1)
            else:
                gen = worker.process_subject((self._random_ids, self._batch_size))
                for _ in range(len(self._random_ids) // self._batch_size):
                    print(_)
                    next(gen)
        except Exception as e:
            print(e)
        '''
        self._ray_workers: List[RayWorker] = [
            RayWorker.remote(self._reader, self._scaler, self._row_only, self._bining,
                             self._columns, self._target_replication)
            for _ in range(self._cpu_count)
        ]

    def _start_epoch(self):
        random.shuffle(self._random_ids)
        ids = self.split_ids(self._random_ids, len(self._ray_workers))
        if self._deep_supervision:
            self.__results = [
                worker.process_subject_deep_supervision.options(num_returns="dynamic").remote(
                    (subject_ids, self._batch_size))
                for worker, subject_ids in zip(self._ray_workers, ids)
            ]
        else:
            self.__results = [
                worker.process_subject.options(num_returns="dynamic").remote(
                    (subject_ids, self._batch_size))
                for worker, subject_ids in zip(self._ray_workers, ids)
            ]

    def __generator(self):
        while True:
            if self._cpu_count:
                ready_ids, _ = ray.wait(self.__results, num_returns=1)
                dynamci_result = ray.get(ready_ids[0])
                for object_result in dynamci_result:
                    X, y, t = ray.get(object_result)
                    yield X, y, t
            else:
                random.shuffle(self._random_ids)
                if self._deep_supervision:
                    for X, y, M in process_subject_deep_supervision(args=(self._random_ids,
                                                                          self._batch_size),
                                                                    reader=self._reader,
                                                                    scaler=self._scaler,
                                                                    bining=self._bining):
                        yield X, y, M
                else:
                    for X, y, t in process_subject(args=(self._random_ids, self._batch_size),
                                                   reader=self._reader,
                                                   scaler=self._scaler,
                                                   row_only=self._row_only,
                                                   bining=self._bining,
                                                   target_replication=self._target_replication):
                        yield X, y, t

    @staticmethod
    def split_ids(input_list, cpu_count):
        chunk_size = len(input_list) // cpu_count
        remainder = len(input_list) % cpu_count

        chunks = []
        start = 0
        for i in range(int(cpu_count)):
            end = int(start + chunk_size + (1 if i < remainder else 0))
            chunks.append(input_list[start:end])
            start = end

        return chunks

    def _close(self):
        try:
            ray.get(self.__results)
            for worker in self._ray_workers:
                worker.exit.remote()
            self._ray_workers.clear()
        except ValueError as e:
            # If shutdown is quicker than this this will
            # raise a ValueError. We can safely ignore this
            pass

    @staticmethod
    def _stack_batches(data):
        max_len = max([x.shape[1] for x in data])
        data = [
            np.concatenate([x, np.zeros([x.shape[0], max_len - x.shape[1], x.shape[2]])],
                           axis=1,
                           dtype=np.float32) if max_len - x.shape[1] else x for x in data
        ]
        return np.concatenate(data, axis=0, dtype=np.float32)

    def _subjects_for_samples(self,
                              target_size: int,
                              max_iter: int = 20) -> Tuple[List[float], int]:
        """
        Selects subjects to match the target number of samples.
        """
        assert self._tracker.subject_ids
        # Init tracking vars
        best_diff = float('inf')
        iter = 0

        # Init subject counts
        subject_df = pd.DataFrame(deepcopy(self._tracker.subjects)).T["total"]

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
                    subject_samples = tracker_pr.subjects[next_subject]['total']

                if current_size + subject_samples <= target_size_pr:
                    current_size += subject_samples
                    subjects.append(next_subject)
                    remaining_subjects.drop(next_subject)

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
        with Pool(n_cpus, initializer=init,
                  initargs=(subject_df, target_size, self._tracker, lock)) as pool:
            # Try max_iter times and fetch best result
            res = pool.imap_unordered(compute_samples,
                                      range(max_iter),
                                      chunksize=int(np.ceil(max_iter / n_cpus)))

            for diff, current_size, subjects in res:
                iter += 1
                # Fetch best result
                if diff < best_diff:
                    best_subjects, best_size, best_diff = subjects, current_size, diff

                # Break if no diff
                if best_diff == 0 or iter >= max_iter:
                    break
            pool.close()
            pool.join()

        # Always get smallest best, so if no best found target size is too small
        if not best_subjects:
            return subject_df.min(), subject_df.argmin()

        return best_subjects, best_size


# TODO! these worker functions must go somewhere else


@ray.remote
class RayWorker:

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler,
                 row_only: bool,
                 bining: str,
                 columns: list,
                 target_replication: bool = False):
        self._reader = reader
        self._scaler = scaler
        self._row_only = row_only
        self._bining = bining
        self._columns = columns
        self._target_replication = target_replication

    def process_subject_deep_supervision(self, args):
        return process_subject_deep_supervision(args,
                                                reader=self._reader,
                                                scaler=self._scaler,
                                                bining=self._bining)

    def process_subject(self, args):
        return process_subject(args,
                               reader=self._reader,
                               scaler=self._scaler,
                               row_only=self._row_only,
                               bining=self._bining,
                               target_replication=self._target_replication)

    def exit(self):
        ray.actor.exit_actor()


def process_subject_deep_supervision(args, reader: ProcessedSetReader, scaler: AbstractScaler,
                                     bining: str):
    # TODO! deep supervision binning
    subject_ids, batch_size = args
    # Store the current logging level
    previous_logging_level = logging.getLogger().level

    # Set logging level to CRITICAL to suppress logging
    logging.getLogger().setLevel(logging.CRITICAL)
    # try:
    X_batch, y_batch, m_batch, t_batch = list(), list(), list(), list()
    for subject_id in subject_ids:
        X_subject, y_subject, M_subject = reader.read_sample(subject_id,
                                                             read_masks=True,
                                                             read_ids=True).values()
        for stay_id in X_subject.keys():
            X_stay = X_subject[stay_id]
            X_stay[X_stay.columns] = scaler.transform(X_stay)
            X_batch.append(X_stay)
            y_batch.append(y_subject[stay_id])
            m_batch.append(M_subject[stay_id])
            if len(X_batch) == batch_size:
                # Shuffle the inside of the batch again
                X_batch, y_batch, m_batch = shuffled_data(X_batch, y_batch, m_batch)
                X = zeropad_samples(X_batch)
                y = zeropad_samples(y_batch)
                m = zeropad_samples(m_batch)
                y = np.array(y)
                m = np.array(m)
                X_batch.clear()
                y_batch.clear()
                m_batch.clear()
                yield X, y, m
    if X_batch:
        # Shuffle the inside of the batch again
        X_batch, y_batch, m_batch = shuffled_data(X_batch, y_batch, m_batch)
        X = zeropad_samples(X_batch)
        y = zeropad_samples(y_batch)
        m = zeropad_samples(m_batch)
        y = np.array(y)
        m = np.array(m)
        X_batch.clear()
        y_batch.clear()
        m_batch.clear()
        t_batch.clear()
        yield X, y, m
    # finally:
    # Restore the previous logging level
    logging.getLogger().setLevel(previous_logging_level)
    return


def process_subject(args, reader: ProcessedSetReader, scaler: AbstractScaler, row_only: bool,
                    bining: str, target_replication: bool):
    subject_ids, batch_size = args
    # Store the current logging level
    previous_logging_level = logging.getLogger().level

    # Set logging level to CRITICAL to suppress logging
    logging.getLogger().setLevel(logging.CRITICAL)
    # try:
    X_batch, y_batch, t_batch = list(), list(), list()
    for subject_id in subject_ids:
        X_subject, y_subject = reader.read_sample(subject_id, read_ids=True).values()
        for stay_id in X_subject.keys():
            X_stay, y_stay = X_subject[stay_id], y_subject[stay_id]
            X_stay[X_stay.columns] = scaler.transform(X_stay)
            Xs, ys, ts = read_timeseries(X_df=X_stay, y_df=y_stay, row_only=row_only, bining=bining)
            Xs, ys, ts = shuffled_data(Xs, ys, ts)
            for X, y, t in zip(Xs, ys, ts):
                if target_replication:
                    y = np.atleast_2d(y).repeat(X.shape[0], axis=0)
                X_batch.append(X)
                y_batch.append(y)
                t_batch.append(t)
                if len(X_batch) == batch_size:
                    # Shuffle the inside of the batch again
                    X_batch, y_batch, t_batch = shuffled_data(X_batch, y_batch, t_batch)
                    X = zeropad_samples(X_batch)
                    if target_replication:
                        y = zeropad_samples(y_batch)
                    else:
                        y = np.array(y_batch)
                    t = np.array(t_batch, dtype=np.float32)
                    X_batch.clear()
                    y_batch.clear()
                    t_batch.clear()
                    yield X, y, t

    if X_batch:
        # Shuffle the inside of the batch again
        X_batch, y_batch, t_batch = shuffled_data(X_batch, y_batch, t_batch)
        X = zeropad_samples(X_batch)
        if target_replication:
            y = zeropad_samples(y_batch)
        else:
            y = np.array(y_batch)
        t = np.array(t_batch, dtype=np.float32)
        X_batch.clear()
        y_batch.clear()
        t_batch.clear()
        yield X, y, t
    #finally:
    # Restore the previous logging level
    logging.getLogger().setLevel(previous_logging_level)
    return


def shuffled_data(Xs, ys, ts):
    indices = list(range(len(Xs)))
    random.shuffle(indices)
    Xs = [Xs[i] for i in indices]
    ys = [ys[i] for i in indices]
    ts = [ts[i] for i in indices]
    return Xs, ys, ts
