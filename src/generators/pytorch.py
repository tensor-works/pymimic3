import torch
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
        super().__init__(dataset=TorchDataset(reader=reader,
                                              scaler=scaler,
                                              batch_size=1,
                                              shuffle=shuffle,
                                              bining=bining),
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        samples, labels = zip(*batch)
        samples = torch.stack(self._zeropad_samples(samples))
        labels = torch.cat(labels)
        if labels.dim() == 1:
            return samples, labels.unsqueeze(1)
        return samples, labels

    @staticmethod
    def _zeropad_samples(data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Ensure data is a list of PyTorch tensors
        if not all(isinstance(x, torch.Tensor) for x in data):
            raise ValueError("All items in the data list must be PyTorch tensors")

        # Determine the dtype and device from the first tensor
        dtype = data[0].dtype
        device = data[0].device

        # Find the maximum length along the first dimension
        max_len = max(x.shape[0] for x in data)

        # Pad each tensor to the maximum length
        padded_data = [
            torch.cat(
                [x,
                 torch.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype, device=device)],
                dim=0) for x in data
        ]

        # Stack all padded tensors into a single tensor
        return padded_data


class TorchDataset(AbstractGenerator, Dataset):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 bining: str = "none"):
        super(TorchDataset, self).__init__(reader=reader,
                                           scaler=scaler,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           bining=bining)

    def __getitem__(self, index=None):
        X, y = super().__getitem__(index)
        X = X.squeeze()
        return torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.int8)
