import pandas as pd
from pathlib import Path
from typing import Dict, Union
from multipledispatch import dispatch
from numbers import Number
from utils.IO import *
from settings import *
from .splitters import ReaderSplitter, CompactSplitter
from ..readers import ProcessedSetReader, SplitSetReader

__all__ = ['train_test_split']


@dispatch(dict,
          dict,
          test_size=float,
          val_size=float,
          train_size=Number,
          demographic_split=dict,
          demographic_filter=dict,
          source_path=Path)
def train_test_split(X_subjects: Dict[str, Dict[str, pd.DataFrame]],
                     y_subjects: Dict[str, Dict[str, pd.DataFrame]],
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     source_path: Path = None) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    return CompactSplitter().split_dict(X_subjects=X_subjects,
                                        y_subjects=y_subjects,
                                        test_size=test_size,
                                        val_size=val_size,
                                        train_size=train_size,
                                        source_path=source_path,
                                        demographic_split=demographic_split,
                                        demographic_filter=demographic_filter)


@dispatch(ProcessedSetReader,
          test_size=float,
          val_size=float,
          train_size=Number,
          demographic_split=dict,
          demographic_filter=dict,
          storage_path=Path)
def train_test_split(reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path=None) -> SplitSetReader:
    return ReaderSplitter().split_reader(reader=reader,
                                         test_size=test_size,
                                         val_size=val_size,
                                         train_size=train_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter,
                                         storage_path=storage_path)


@dispatch(ProcessedSetReader,
          float,
          val_size=float,
          train_size=Number,
          demographic_split=dict,
          demographic_filter=dict,
          storage_path=Path)
def train_test_split(reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path=None) -> SplitSetReader:
    return ReaderSplitter().split_reader(reader=reader,
                                         test_size=test_size,
                                         val_size=val_size,
                                         train_size=train_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter,
                                         storage_path=storage_path)


@dispatch(ProcessedSetReader,
          float,
          float,
          train_size=Number,
          demographic_split=dict,
          demographic_filter=dict,
          storage_path=Path)
def train_test_split(reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path=None) -> SplitSetReader:
    return ReaderSplitter().split_reader(reader=reader,
                                         test_size=test_size,
                                         val_size=val_size,
                                         train_size=train_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter,
                                         storage_path=storage_path)


@dispatch(ProcessedSetReader,
          float,
          float,
          Number,
          demographic_split=dict,
          demographic_filter=dict,
          storage_path=Path)
def train_test_split(reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path=None) -> SplitSetReader:
    return ReaderSplitter().split_reader(reader=reader,
                                         test_size=test_size,
                                         val_size=val_size,
                                         train_size=train_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter,
                                         storage_path=storage_path)


@dispatch(ProcessedSetReader,
          float,
          float,
          Number,
          dict,
          demographic_filter=dict,
          storage_path=Path)
def train_test_split(reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path=None) -> SplitSetReader:
    return ReaderSplitter().split_reader(reader=reader,
                                         test_size=test_size,
                                         val_size=val_size,
                                         train_size=train_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter,
                                         storage_path=storage_path)


@dispatch(ProcessedSetReader, float, float, Number, dict, dict, storage_path=Path)
def train_test_split(reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path=None) -> SplitSetReader:
    return ReaderSplitter().split_reader(reader=reader,
                                         test_size=test_size,
                                         val_size=val_size,
                                         train_size=train_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter,
                                         storage_path=storage_path)


@dispatch(ProcessedSetReader, float, float, dict, dict, Path)
def train_test_split(reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     train_size: int = None,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path=None) -> SplitSetReader:
    return ReaderSplitter().split_reader(reader=reader,
                                         test_size=test_size,
                                         val_size=val_size,
                                         train_size=train_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter,
                                         storage_path=storage_path)
