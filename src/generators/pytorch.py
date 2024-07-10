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
                 shuffle: bool = True,
                 n_samples: int = None,
                 num_cpus: int = 0,
                 target_replication: bool = False,
                 deep_supervision: bool = False,
                 drop_last: bool = False,
                 bining: str = "none"):
        self._dataset = TorchDataset(reader=reader,
                                     scaler=scaler,
                                     num_cpus=num_cpus,
                                     n_samples=n_samples,
                                     batch_size=1,
                                     deep_supervision=deep_supervision,
                                     target_replication=target_replication,
                                     shuffle=shuffle,
                                     bining=bining)
        super().__init__(dataset=self._dataset,
                         batch_size=1,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=0,
                         collate_fn=self.collate_fn)
        self._deep_supervision = deep_supervision

    def collate_fn(self, batch):
        if self._deep_supervision:
            samples, labels, masks = zip(*batch)
            masks = masks[0]
            if masks.dim() == 1:
                masks = masks.unsqueeze(1)
        else:
            samples, labels = zip(*batch)
        samples, labels = samples[0], labels[0]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        if self._deep_supervision:
            return [samples, masks], labels
        return samples, labels

    def close(self):
        self._dataset.__del__()


class TorchDataset(AbstractGenerator, Dataset):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 num_cpus: int = 0,
                 n_samples: int = None,
                 deep_supervision: bool = False,
                 target_replication: bool = False,
                 shuffle: bool = True,
                 bining: str = "none"):
        AbstractGenerator.__init__(self,
                                   reader=reader,
                                   scaler=scaler,
                                   num_cpus=num_cpus,
                                   n_samples=n_samples,
                                   batch_size=batch_size,
                                   deep_supervision=deep_supervision,
                                   target_replication=target_replication,
                                   shuffle=shuffle,
                                   bining=bining)

    def __getitem__(self, index=None):
        if self._deep_supervision:
            X, y, m = super().__getitem__(index)
            if not m.flags.writeable:
                m = m.copy()
        else:
            X, y = super().__getitem__(index)
        if not X.flags.writeable:
            X = X.copy()
        if not y.flags.writeable:
            y = y.copy()
        if self._deep_supervision:
            return torch.from_numpy(X).to(torch.float32), \
                   torch.from_numpy(y), \
                   torch.from_numpy(m)
        return torch.from_numpy(X).to(torch.float32), torch.from_numpy(y)

    def close(self):
        AbstractGenerator.__del__(self)
