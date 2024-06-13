"""
This module provides functionality to split datasets into training, validation, and test sets.
Splits can be performed based on predefined ratios or demographic filters, and the module
supports both dictionary-based and reader-based dataset structures.


Examples
--------
>>> # Split a reader by ratio
>>> reader = ProcessedSetReader(root_path="/path/to/data")
>>> split_reader = ReaderSplitter().split_reader(reader, test_size=0.2, val_size=0.1)
>>> split_reader.test.subject_ids
>>> [ ... ]

>>> # Split a reader by demographics with ratio control
>>> demographic_split = {
...     "test": {
...         "AGE": {
...             "greater": 60
...         }
...     },
...     "val": {
...         "AGE": {
...             "leq": 60,
...             "greater": 40
...         }
...     }
... }
>>> split_reader = ReaderSplitter().split_reader(reader, 
...                                              test_size=0.2, 
...                                              val_size=0.1, 
...                                              demographic_split=demographic_split)
>>> split_reader.test.subject_ids
>>> [ ... ]

>>> # Or split a reader by demographics only
>>> demographic_split = {
...     "test": {
...         "AGE": {
...             "greater": 60
...         }
...     },
...     "val": {
...         "AGE": {
...             "leq": 60,
...             "greater": 40
...         }
...     }
... }
>>> split_reader = ReaderSplitter().split_reader(demographic_split=demographic_split)

>>> # Reduce a reader to a subdemographic and split by ratio
>>> demographic_filter = {
...     "AGE": {
...         "greater": 60
...     }
... }
>>> split_reader = ReaderSplitter().split_reader(reader, 
...                                              test_size=0.2, 
...                                              val_size=0.1, 
...                                              demographic_split=demographic_split)

>>> # Demographic split with categorical attribute
>>> demographic_split = {
...     "test": {
...         "GENDER": {
...             "choice": ["M"]
...         }
...     },
...     "val": {
...         "GENDER": {
...             "choice": ["F"]
...         }
...     }
... }
>>> split_reader = ReaderSplitter().split_reader(reader, 
...                                              test_size=0.2, 
...                                              val_size=0.1, 
...                                              demographic_split=demographic_split)

"""

import pandas as pd
from pathlib import Path
from typing import Dict, Union
from multipledispatch import dispatch
from numbers import Number
from utils.IO import *
from settings import *
from .splitters import ReaderSplitter, CompactSplitter
from ..readers import ProcessedSetReader, SplitSetReader


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
    """
    Splits the dictionary-based dataset into training, validation, and test sets.

    Parameters
    ----------
    X_subjects : dict
        Dictionary containing feature data for subjects.
    y_subjects : dict
        Dictionary containing label data for subjects.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Default is 0.0.
    val_size : float, optional
        Proportion of the dataset to include in the validation split. Default is 0.0.
    train_size : int, optional
        Number of samples to include in the training split. If specified, overrides the proportion-based split for the training set. Default is None.
    demographic_split : dict, optional
        Dictionary specifying demographic criteria for splitting the dataset. Default is None.
    demographic_filter : dict, optional
        Dictionary specifying demographic criteria for filtering the dataset before splitting. Default is None.
    source_path : Path, optional
        Path to the directory containing the dataset. Default is None.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        Dictionary containing split data for training, validation, and test sets.
    """
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
    """
    Splits the reader-based dataset into training, validation, and test sets.

    Parameters
    ----------
    reader : ProcessedSetReader
        Reader object to load the dataset.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Default is 0.0.
    val_size : float, optional
        Proportion of the dataset to include in the validation split. Default is 0.0.
    train_size : int, optional
        Number of samples to include in the training split. If specified, overrides the proportion-based split for the training set. Default is None.
    demographic_split : dict, optional
        Dictionary specifying demographic criteria for splitting the dataset. Default is None.
    demographic_filter : dict, optional
        Dictionary specifying demographic criteria for filtering the dataset before splitting. Default is None.
    storage_path : Path, optional
        Path to the directory where the split information will be saved. Default is None, which uses the reader's root path.

    Returns
    -------
    SplitSetReader
        Reader object for the split dataset.
    """
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
