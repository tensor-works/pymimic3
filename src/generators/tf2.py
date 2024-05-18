from preprocessing.scalers import AbstractScaler
from datasets.readers import ProcessedSetReader
from tensorflow.keras.utils import Sequence
from tests.settings import *
from utils.IO import *
from . import AbstractGenerator


class TFGenerator(AbstractGenerator, Sequence):

    def __init__(self,
                 reader: ProcessedSetReader,
                 scaler: AbstractScaler = None,
                 batch_size: int = 8,
                 shuffle: bool = True):
        super(TFGenerator, self).__init__(reader=reader,
                                          scaler=scaler,
                                          batch_size=batch_size,
                                          shuffle=shuffle)

    def __getitem__(self, index=None):
        return super().__getitem__(index)
