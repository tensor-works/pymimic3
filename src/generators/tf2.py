from preprocessing.scalers import AbstractScaler
from datasets.readers import ProcessedSetReader
from tensorflow.keras.utils import Sequence
from utils.IO import *
from . import AbstractGenerator


class TFGenerator(AbstractGenerator, Sequence):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 num_cpus: int = 0,
                 deep_supervision: bool = False,
                 target_replication: bool = False,
                 shuffle: bool = True,
                 bining: str = "none"):
        AbstractGenerator.__init__(self,
                                   reader=reader,
                                   scaler=scaler,
                                   num_cpus=num_cpus,
                                   batch_size=batch_size,
                                   target_replication=target_replication,
                                   deep_supervision=deep_supervision,
                                   shuffle=shuffle,
                                   bining=bining)
        self._deep_supervision = deep_supervision

    def __getitem__(self, index=None):
        if self._deep_supervision:
            X, y, m = super().__getitem__(index)
            if len(m.shape) == 1:
                m = m.reshape(-1, 1)
        else:
            X, y = super().__getitem__(index)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if self._deep_supervision:
            return (X, m), y
        return X, y
