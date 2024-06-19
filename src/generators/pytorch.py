import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from preprocessing.scalers import AbstractScaler
from datasets.readers import ProcessedSetReader
from torch.utils.data import Dataset
from utils.IO import *
from torch.utils.data import DataLoader
from . import AbstractGenerator


class TorchGenerator(DataLoader):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 1,
                 drop_last: bool = False,
                 bining: str = "none"):
        self._dataset = TorchDataset(reader=reader,
                                     scaler=scaler,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     bining=bining)
        super().__init__(dataset=self._dataset,
                         batch_size=1,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=0,
                         collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        samples, labels = zip(*batch)
        samples, labels = samples[0], labels[0]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return samples, labels

    def __iter__(self) -> _BaseDataLoaderIter:
        return super().__iter__()

    def __next__(self):
        return super().__next__()

    def close(self):
        self._dataset.__del__()


class TorchDataset(AbstractGenerator, Dataset):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 bining: str = "none"):
        AbstractGenerator.__init__(self,
                                   reader=reader,
                                   scaler=scaler,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   bining=bining)

    def __getitem__(self, index=None):
        X, y = super().__getitem__(index)
        if not X.flags.writeable:
            X = X.copy()

        if not y.flags.writeable:
            y = y.copy()
        return torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.float32)

    def close(self):
        AbstractGenerator.__del__(self)
