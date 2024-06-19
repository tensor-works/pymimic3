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
                 num_cpu: int = None,
                 shuffle: bool = True,
                 bining: str = "none"):
        AbstractGenerator.__init__(self,
                                   reader=reader,
                                   scaler=scaler,
                                   num_cpus=num_cpu,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   bining=bining)

    def __getitem__(self, index=None):
        X, y = super().__getitem__(index)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return X, y
