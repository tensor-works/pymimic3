import numpy as np
import pandas as pd
import random
import ray
import os
import logging

from copy import deepcopy
from pathlib import Path
from utils.IO import *
from preprocessing.scalers import AbstractScaler
from datasets.trackers import PreprocessingTracker
from datasets.readers import ProcessedSetReader
from metrics import CustomBins, LogBins
from pathos import multiprocessing as mp


class AbstractGenerator:

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler,
                 num_cpus: int = None,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 bining: str = "none",
                 deep_supervision: bool = False):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._reader = reader
        self._columns = None
        self._deep_supervision = deep_supervision
        self._tracker = PreprocessingTracker(storage_path=Path(reader.root_path, "progress"))
        self._steps = self._count_batches()
        self._subject_ids = reader.subject_ids
        self._scaler = scaler
        self._random_ids = deepcopy(self._reader.subject_ids)
        random.shuffle(self._random_ids)
        self.generator = self._generator()
        self._row_only = False
        if num_cpus is None:
            self._cpu_count = max(1, len(self._subject_ids) // (mp.cpu_count() - 2))
        else:
            self._cpu_count = min(num_cpus, mp.cpu_count())
        self._workers = list()
        if bining not in ["none", "log", "custom"]:
            raise ValueError("Bining must be one of ['none', 'log', 'custom']")
        self._bining = bining
        self._counter = 0
        self._remainder_X = np.array([])
        self._remainder_y = np.array([])
        self._remainder_M = np.array([])

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=self._cpu_count)
            log_dir = os.path.join(ray._private.utils.get_user_temp_dir(),
                                   "ray/session_latest/logs")
            print(f"Ray logs can be found in: {log_dir}")

    def __getitem__(self, index=None):
        if not self._workers:
            self._create_workers()
            self._start_epoch()

        # Start with any remainder from the previous batch
        X, y, m = next(self.generator)  # if not deepsupervsion m is timestamps else mask
        # Fetch new data until we have at least the required batch size
        while X.shape[0] < self._batch_size:
            X_res = self._remainder_X
            y_res = self._remainder_y
            X = self._stack_batches((X, X_res)) if X_res.size else X
            if self._deep_supervision:
                m_res = self._remainder_M
                m = self._stack_batches((m, m_res)) if m_res.size else m
                y = self._stack_batches((y, y_res)) if y_res.size else y
            else:
                y = np.concatenate((y, y_res), axis=0, dtype=np.float32) if y_res.size else y
            if X.shape[0] < self._batch_size:
                self._remainder_X, self._remainder_y, self._remainder_M = next(self.generator)

            # If the accumulated batch is larger than required, split it
            if X.shape[0] > self._batch_size:
                self._remainder_X = X[self._batch_size:, :, :]
                self._remainder_y = y[self._batch_size:]
                X = X[:self._batch_size]
                y = y[:self._batch_size]
                if self._deep_supervision:
                    self._remainder_M = m[self._batch_size:]
                    m = m[:self._batch_size]

                break

        self._counter += 1
        if self._counter >= self._steps:
            self._close()
            self._counter = 0
            self._remainder_X = np.array([])
            self._remainder_y = np.array([])
            self._remainder_m = np.array([])

        if self._deep_supervision:
            return X, y, m
        return X, y

    def _count_batches(self):
        if self._deep_supervision:
            return max(len(self._tracker.stay_ids) // self._batch_size, 1)
        return sum([
            self._tracker.subjects[subject_id]["total"] for subject_id in self._reader.subject_ids
        ]) // self._batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self._steps

    def __del__(self):
        self._close()

    def _create_workers(self):
        # worker = RayWorker(self._reader, self._scaler, self._row_only, self._bining, self._columns)
        # gen = worker.process_subject((self._random_ids, self._batch_size))
        # X, y, t = next(gen)
        self._workers = [
            RayWorker.remote(self._reader, self._scaler, self._row_only, self._bining,
                             self._columns, self._deep_supervision) for _ in range(self._cpu_count)
        ]

    def _start_epoch(self):
        random.shuffle(self._random_ids)
        ids = self.split_ids(self._random_ids, len(self._workers))
        if self._deep_supervision:
            self.results = [
                worker.process_subject_deep_supervision.options(num_returns="dynamic").remote(
                    (subject_ids, self._batch_size))
                for worker, subject_ids in zip(self._workers, ids)
            ]
        else:
            self.results = [
                worker.process_subject.options(num_returns="dynamic").remote(
                    (subject_ids, self._batch_size))
                for worker, subject_ids in zip(self._workers, ids)
            ]

    def _generator(self):
        while True:
            ready_ids, _ = ray.wait(self.results, num_returns=1)
            dynamci_result = ray.get(ready_ids[0])
            # self.results.remove(ready_ids[0])
            for object_result in dynamci_result:
                X, y, t = ray.get(object_result)
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

    @staticmethod
    def read_timeseries(X_df: pd.DataFrame, y_df: pd.DataFrame, row_only=False, bining="none"):
        if bining == "log":
            y = y_df.applymap(LogBins.get_bin_log)
        elif bining == "custom":
            y = y_df.applymap(CustomBins.get_bin_custom)
        else:
            y = y_df

        if row_only:
            Xs = [X_df.loc[timestamp].values for timestamp in y_df.index]
        else:
            Xs = [X_df.loc[:timestamp].values for timestamp in y_df.index]

        indices = np.random.permutation(len(Xs))
        ys = y.squeeze(axis=1).values.tolist()
        ts = y_df.index.tolist()

        return Xs, ys, ts

    @staticmethod
    def _shuffled_data(Xs, ys, ts):
        indices = list(range(len(Xs)))
        random.shuffle(indices)
        Xs = [Xs[i] for i in indices]
        ys = [ys[i] for i in indices]
        ts = [ts[i] for i in indices]
        return Xs, ys, ts

    def _close(self):
        for worker in self._workers:
            ray.kill(worker)
        self._workers.clear()

    @staticmethod
    def _zeropad_samples(data):
        max_len = max([x.shape[0] for x in data])
        ret = [
            np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:])],
                           axis=0,
                           dtype=np.float32) for x in data
        ]
        return np.atleast_3d(np.array(ret, dtype=np.float32))

    @staticmethod
    def _stack_batches(data):
        max_len = max([x.shape[1] for x in data])
        data = [
            np.concatenate([x, np.zeros([x.shape[0], max_len - x.shape[1], x.shape[2]])],
                           axis=1,
                           dtype=np.float32) if max_len - x.shape[1] else x for x in data
        ]
        return np.concatenate(data, axis=0, dtype=np.float32)


@ray.remote
class RayWorker:

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler,
                 row_only: bool,
                 bining: str,
                 columns: list,
                 deep_supervision: bool = True):
        self._reader = reader
        self._scaler = scaler
        self._row_only = row_only
        self._bining = bining
        self._columns = columns
        self._deep_supervision = deep_supervision

    def process_subject_deep_supervision(self, args):
        subject_ids, batch_size = args
        # Store the current logging level
        previous_logging_level = logging.getLogger().level

        # Set logging level to CRITICAL to suppress logging
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            X_batch, y_batch, m_batch, t_batch = list(), list(), list(), list()
            for subject_id in subject_ids:
                if self._deep_supervision:
                    X_subject, y_subject, M_subject = self._reader.read_sample(
                        subject_id, read_masks=True, read_ids=True).values()
                    for stay_id in X_subject.keys():
                        X_batch.append(X_subject[stay_id])
                        y_batch.append(y_subject[stay_id])
                        m_batch.append(M_subject[stay_id])
                        if len(X_batch) == batch_size:
                            X = AbstractGenerator._zeropad_samples(X_batch)
                            y = AbstractGenerator._zeropad_samples(y_batch)
                            m = AbstractGenerator._zeropad_samples(m_batch)
                            y = np.array(y, dtype=np.float32)
                            m = np.array(m, dtype=np.float32)
                            X_batch.clear()
                            y_batch.clear()
                            m_batch.clear()
                            yield X, y, m
            if X_batch:
                X = AbstractGenerator._zeropad_samples(X_batch)
                y = AbstractGenerator._zeropad_samples(y_batch)
                m = AbstractGenerator._zeropad_samples(m_batch)
                y = np.array(y, dtype=np.float32)
                m = np.array(m, dtype=np.float32)
                X_batch.clear()
                y_batch.clear()
                m_batch.clear()
                t_batch.clear()
                yield X, y, m
        finally:
            # Restore the previous logging level
            logging.getLogger().setLevel(previous_logging_level)
        return

    def process_subject(self, args):
        subject_ids, batch_size = args
        # Store the current logging level
        previous_logging_level = logging.getLogger().level

        # Set logging level to CRITICAL to suppress logging
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            X_batch, y_batch, t_batch = list(), list(), list()
            for subject_id in subject_ids:
                X_subject, y_subject = self._reader.read_sample(subject_id, read_ids=True).values()
                for stay_id in X_subject.keys():
                    X_stay, y_stay = X_subject[stay_id], y_subject[stay_id]
                    X_stay[X_stay.columns] = self._scaler.transform(X_stay)
                    Xs, ys, ts = AbstractGenerator.read_timeseries(X_df=X_stay,
                                                                   y_df=y_stay,
                                                                   row_only=self._row_only,
                                                                   bining=self._bining)
                    Xs, ys, ts = AbstractGenerator._shuffled_data(Xs, ys, ts)
                    for X, y, t in zip(Xs, ys, ts):
                        X_batch.append(X)
                        y_batch.append(y)
                        t_batch.append(t)
                        if len(X_batch) == batch_size:
                            X = AbstractGenerator._zeropad_samples(X_batch)
                            y = np.array(y_batch, dtype=np.float32)
                            t = np.array(t_batch, dtype=np.float32)
                            X_batch.clear()
                            y_batch.clear()
                            t_batch.clear()
                            yield X, y, t

            if X_batch:
                X = AbstractGenerator._zeropad_samples(X_batch)
                y = np.array(y_batch, dtype=np.float32)
                t = np.array(t_batch, dtype=np.float32)
                X_batch.clear()
                y_batch.clear()
                t_batch.clear()
                yield X, y, t
        finally:
            # Restore the previous logging level
            logging.getLogger().setLevel(previous_logging_level)
        return
