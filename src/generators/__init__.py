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


@ray.remote
class RayWorker:

    def __init__(self, reader, scaler, row_only, bining, columns):
        self.reader = reader
        self.scaler = scaler
        self.row_only = row_only
        self.bining = bining
        self.columns = columns

    def process_subject(self, subject_ids):
        # Store the current logging level
        previous_logging_level = logging.getLogger().level

        # Set logging level to CRITICAL to suppress logging
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            result = []
            for subject_id in subject_ids:
                X_subject, y_subject = self.reader.read_sample(subject_id).values()
                for X_stay, y_stay in zip(X_subject, y_subject):
                    X_stay[X_stay.columns] = self.scaler.transform(X_stay)
                    Xs, ys, ts = AbstractGenerator.read_timeseries(X_df=X_stay,
                                                                   y_df=y_stay,
                                                                   row_only=self.row_only,
                                                                   bining=self.bining)
                    Xs, ys, ts = AbstractGenerator._shuffled_data(Xs, ys, ts)
                    for X, y, t in zip(Xs, ys, ts):
                        result.append((X, y, t))
        finally:
            # Restore the previous logging level
            logging.getLogger().setLevel(previous_logging_level)
        return result


class AbstractGenerator:

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 bining: str = "none"):
        super().__init__()
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._reader = reader
        self._columns = None
        self._tracker = PreprocessingTracker(storage_path=Path(reader.root_path, "progress"))
        self._steps = self._count_batches()
        self._subject_ids = reader.subject_ids
        self._scaler = scaler
        self._random_ids = deepcopy(self._reader.subject_ids)
        random.shuffle(self._random_ids)
        self.generator = self._generator()
        self._row_only = False
        self._cpu_count = max(1, len(self._subject_ids) // (mp.cpu_count() - 2))

        if bining not in ["none", "log", "custom"]:
            raise ValueError("Bining must be one of ['none', 'log', 'custom']")
        self._bining = bining

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=self._cpu_count)
            log_dir = os.path.join(ray._private.utils.get_user_temp_dir(),
                                   "ray/session_latest/logs")
            print(f"Ray logs can be found in: {log_dir}")

        # Initialize Ray actors
        self._workers = [
            RayWorker.remote(reader, scaler, self._row_only, bining, self._columns)
            for _ in range(self._cpu_count)
        ]

    @property
    def steps(self):
        return self._steps

    def __getitem__(self, index=None):
        X_batch, y_batch = list(), list()
        for _ in range(self._batch_size):
            X, y = next(self.generator)
            X_batch.append(X)
            y_batch.append(y)
        X_batch = self._zeropad_samples(X_batch)
        y_batch = np.array(y_batch)
        return X_batch.astype(np.float32), y_batch.astype(np.float32)

    def _count_batches(self):
        return int(
            np.floor(
                sum([
                    self._tracker.subjects[subject_id]["total"]
                    for subject_id in self._reader.subject_ids
                ])) / self._batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self._steps

    def __del__(self):
        self._close()

    def on_epoch_end(self):
        ...

    def _generator(self):

        def start_epoch():
            random.shuffle(self._random_ids)
            ids = self.split_ids(self._random_ids, len(self._workers))
            self.results = [
                worker.process_subject.remote(subject_ids)
                for worker, subject_ids in zip(self._workers, ids)
            ]

        start_epoch()
        finished_count = 0
        while True:
            ready_ids, _ = ray.wait(self.results, num_returns=1)
            result = ray.get(ready_ids[0])
            self.results.remove(ready_ids[0])
            if not result:
                finished_count += 1
            else:
                for X, y, t in result:
                    yield X, y

            if finished_count == self._cpu_count:
                start_epoch()
                print("FINISHED\nFINISHED\nFINISHED")
                finished_count = 0

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

    @staticmethod
    def _zeropad_samples(data):
        dtype = data[0].dtype
        max_len = max([x.shape[0] for x in data])
        ret = [
            np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)],
                           axis=0) for x in data
        ]
        return np.array(ret)

    def _close(self):
        for worker in self._workers:
            ray.kill(worker)


# Example usage
if __name__ == "__main__":
    # Mocking the necessary components for demonstration
    class MockReader:
        subject_ids = [1, 2, 3, 4]
        root_path = "."

        def read_sample(self, subject_id):
            return {
                'X': pd.DataFrame(np.random.rand(100, 10)),
                'y': pd.DataFrame(np.random.rand(100, 1))
            }

    class MockScaler:

        def transform(self, X):
            return (X - np.mean(X)) / np.std(X)

    reader = MockReader()
    scaler = MockScaler()
    generator = AbstractGenerator(reader, scaler)

    for X, y in generator:
        print(X, y)
        break
