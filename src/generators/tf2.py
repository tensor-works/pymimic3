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
                 shuffle: bool = True,
                 bining: str = "none"):
        super(TFGenerator, self).__init__(reader=reader,
                                          scaler=scaler,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          bining=bining)

    def __getitem__(self, index=None):
        X, y = super().__getitem__(index)
        return X, y.reshape(-1, 1)
