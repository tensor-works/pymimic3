import pandas as pd
import numpy as np
from datetime import timedelta
from utils.IO import *
from utils import load_json
from utils import check_data, check_nan
from generators import AbstractGenerator, AbstractBatchGenerator


class BatchReader(AbstractGenerator):

    def __init__(self,
                 reader: object,
                 discretizer: object,
                 type: str,
                 normalizer: object = None,
                 batch_size: int = 8,
                 subset_size: int = None,
                 shuffle: bool = True):
        """_summary_

        Args:
            discretizer (object): _description_
            type (str): _description_
            source_path (Path, optional): _description_. Defaults to None.
            normalizer (object, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 8.
            subset_size (int, optional): _description_. Defaults to None.
            shuffle (bool, optional): _description_. Defaults to True.
        """
        super().__init__(discretizer, type, normalizer, batch_size, subset_size, shuffle)
        # TODO! batch count from progress.json
        self.reader = reader
        self.steps = self.make_batch_count(self.batch_size)
        self.subject_ids = reader.subjects

    def _generator(self):

        while True:
            outer_index = 0
            self.reset_flag = False

            while outer_index < len(self.subject_ids):
                if self.reset_flag:
                    break

                data, subject_folder = self.reader.get_sample()
                X_subject, y_subject = data["X"], data["y"]
                if X_subject is None or y_subject is None:
                    continue

                X_subject, y_subject = check_data(X_subject, y_subject)

                if X_subject is None or y_subject is None:
                    continue

                itterator = zip(X_subject, y_subject)
                outer_index += 1
                for X_ts, y_ts in itterator:

                    X_ts = self.discretize_frames(X_ts)

                    if not len(X_ts) or not len(y_ts):
                        continue

                    X_ts = self.normalize_frames(X_ts)

                    if self.subset_size:
                        remaining = len(y_ts) - self.subset_size
                    else:
                        remaining = len(y_ts)

                    while remaining > 0:
                        if self.reset_flag:
                            break
                        if not len(y_ts):
                            continue
                        if self.subset_size:
                            delimiter = len(y_ts) - min(0, remaining)
                            Xs, ys, ts = self.read_timeseries(X_ts[:delimiter], y_ts[:delimiter])
                            remaining -= self.subset_size
                        else:
                            Xs, ys, ts = self.read_timeseries(X_ts, y_ts)
                            remaining = 0

                        Xs = self._process_frames(Xs, ts)
                        (Xs, ys, ts) = self._shuffled_data([Xs, ys, ts])

                        current_size = len(Xs)
                        itterator = range(0, current_size, self.batch_size)

                        batch_count = 0
                        for i in itterator:
                            if self.reset_flag:
                                break
                            X = self._zeropad_samples(Xs[i:i + self.batch_size]).astype(float)
                            y = np.array(ys[i:i + self.batch_size]).astype(self.output_type)
                            X, y = check_data(X, y)
                            if X is None or y is None:
                                continue
                            batch_data = (np.array(X), np.array(y))
                            check_nan(X, y)

                            batch_count += 1

                            self.batch_data = batch_data

                            yield batch_data

    def make_batch_count(self, batch_size):
        """
        """
        self.progress = load_json(Path(self.reader.root_path, "progress.json"))
        if "finished" in self.progress:
            del self.progress["finished"]
        return int(
            np.array([
                np.ceil(data["total"] / batch_size)
                for id, data in self.progress["subjects"].items()
            ]).sum()) - 1


class MIMICBatchGenerator(AbstractGenerator):
    """
    """

    def __init__(self,
                 X,
                 y,
                 discretizer: object,
                 type: str,
                 normalizer: object = None,
                 batch_size: int = 8,
                 subset_size: int = None,
                 shuffle: bool = True):
        """_summary_

        Args:
            X (list): _description_
            y (list): _description_
            discretizer (object): _description_
            type (str): _description_
            normalizer (object, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 8.
            subset_size (int, optional): _description_. Defaults to None.
            shuffle (bool, optional): _description_. Defaults to True.
        """
        self.X, self.y = check_data(X, y)
        super().__init__(discretizer, type, normalizer, batch_size, subset_size, shuffle)
        self.steps = self.make_batch_count(y, self.batch_size)

    def _generator(self):
        """
        """
        while True:
            outer_index = 0
            self.reset_flag = False

            while outer_index < len(self.y):
                if self.reset_flag:
                    break
                X_ts = self.discretize_frames(self.X[outer_index])

                y_ts = self.y[outer_index]

                X_ts = self.normalize_frames(X_ts)
                if self.subset_size:
                    remaining = len(y_ts) - self.subset_size
                else:
                    remaining = len(y_ts)

                outer_index += 1

                while remaining > 0:
                    if self.reset_flag:
                        break
                    if not len(y_ts):
                        continue
                    if self.subset_size:
                        delimiter = len(y_ts) - min(0, remaining)
                        Xs, ys, ts = self.read_timeseries(X_ts[:delimiter], y_ts[:delimiter])
                        remaining -= self.subset_size
                    else:
                        Xs, ys, ts = self.read_timeseries(X_ts, y_ts)
                        remaining = 0

                    Xs = self._process_frames(Xs, ts)
                    (Xs, ys, ts) = self._shuffled_data([Xs, ys, ts])

                    current_size = len(Xs)
                    itterator = range(0, current_size, self.batch_size)

                    for i in itterator:
                        if self.reset_flag:
                            break
                        X = self._zeropad_samples(Xs[i:i + self.batch_size]).astype(float)
                        y = np.array(ys[i:i + self.batch_size]).astype(self.output_type)
                        batch_data = (X, y)
                        yield batch_data

    def make_batch_count(self, y_df, batch_size):
        """
        """
        return int(np.array([np.ceil(len(ys) / batch_size) for ys in y_df]).sum()) - 1
