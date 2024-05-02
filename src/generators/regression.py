import numpy as np
from utils.IO import *
from utils import check_data
from generators import AbstractGenerator
from pathos.multiprocessing import cpu_count
from pathos.pools import ThreadPool  # ProcessPool


class MIMICBatchReader(AbstractGenerator):

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
        self.reader = reader
        self.subject_ids = reader.subjects
        self.steps = len(self.subject_ids)
        super().__init__(discretizer, type, normalizer, batch_size, subset_size, shuffle)

    def _generator(self):
        """_summary_
        """

        def parallel_process(reader, discretizer, normalizer, subject_folder):
            X_subject, y_subject = reader.read_sample(subject_folder).values()
            if X_subject is None or y_subject is None:
                return None, None

            X_subject, y_subject = check_data(X_subject, y_subject)

            if X_subject is None or y_subject is None:
                return None, None

            itterator = zip(X_subject, y_subject)
            rets = list()
            X_cache = list()
            y_cache = list()

            for X_ts, y_ts in itterator:
                X = discretizer.transform(X_ts)
                X = normalizer.transform(X)
                # X, y = check_data(X, y_ts)

                if not len(X) or not len(y_ts):
                    continue
                X, y = np.stack(X), np.array(y_ts).reshape(len(y_ts), 1)

                X_cache.append(X)
                y_cache.append(y)

            return X_cache, y_cache

        while True:
            pool = ThreadPool(cpu_count() - 1)
            from functools import partial
            partial(parallel_process, self.reader, self.discretizer,
                    self.normalizer)(self.reader.subject_folders[0])
            res = pool.uimap(partial(parallel_process, self.reader, self.discretizer,
                                     self.normalizer),
                             self.reader.subject_folders,
                             chunksize=50)

            X_cache = list()
            y_cache = list()
            for index, (X, y) in enumerate(res):
                if X is None or y is None:
                    continue
                for sample, label in zip(list(X), list(y)):
                    if sample.shape[0] > 1:
                        X_splits = np.split(sample, range(1, len(sample) - 1))
                        y_splits = np.split(label, range(1, len(label) - 1))
                        for split_sample, split_label in zip(list(X_splits), list(y_splits)):
                            X_cache.append(split_sample)
                            y_cache.append(split_label)
                        if len(X_cache) % 10 == 0 and not len(X_cache) == 0:
                            yield np.squeeze(np.concatenate(X_cache)), np.squeeze(
                                np.concatenate(y_cache))
                            X_cache = list()
                            y_cache = list()
                    else:
                        X_cache.append(sample)
                        y_cache.append(label)
                    if len(X_cache) % 10 == 0 and not len(X_cache) == 0:
                        yield np.squeeze(np.concatenate(X_cache)), np.squeeze(
                            np.concatenate(y_cache, axis=1)).T
                        X_cache = list()
                        y_cache = list()
