import torch
import numpy as np
from preprocessing.scalers import AbstractScaler
from datasets.readers import ProcessedSetReader
from torch.utils.data import Dataset
from tests.settings import *
from utils.IO import *
from torch.utils.data import DataLoader
from . import AbstractGenerator


class ScikitGenerator(DataLoader):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 1,
                 drop_last: bool = False):
        super().__init__(dataset=ScikitDataset(
            reader=reader,
            scaler=scaler,
            batch_size=1,
            shuffle=shuffle,
        ),
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        samples, labels = zip(*batch)
        return self._zeropad_samples(list(samples)), np.array(labels)

    @staticmethod
    def _zeropad_samples(data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        dtype = data[0].dtype
        max_len = max([x.shape[0] for x in data])
        ret = [
            np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)],
                           axis=0) for x in data
        ]
        return np.array(ret)


class ScikitDataset(AbstractGenerator, Dataset):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 shuffle: bool = True):
        super(ScikitDataset, self).__init__(reader=reader,
                                            scaler=scaler,
                                            batch_size=batch_size,
                                            shuffle=shuffle)

    def __getitem__(self, index=None):
        X, y = super().__getitem__(index)
        X = X.squeeze()
        return X, y
