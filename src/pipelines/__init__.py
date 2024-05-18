from typing import Union
from pathlib import Path
from generators.tf2 import TFGenerator
from preprocessing.scalers import AbstractScaler, MIMICMinMaxScaler, MIMICStandardScaler, MIMICMaxAbsScaler, MIMICRobustScaler
from datasets.readers import ProcessedSetReader
from datasets.readers import ProcessedSetReader, SplitSetReader
from abc import ABC, abstractmethod
from tests.settings import *
from utils.IO import *


class AbstractPipeline(ABC):

    def __init__(
        self,
        storage_path: Path,
        reader: Union[ProcessedSetReader, SplitSetReader],
        model,
        generator_options: dict,
        model_options: dict = {},
        scaler_options: dict = {},
        compile_options: dict = {},
        data_split_options: dict = {},
        scaler: AbstractScaler = None,
        scaler_type: str = "minmax",
    ):
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._reader = reader
        self._model = model
        self._generator_options = generator_options
        self._data_split_options = data_split_options
        self._split_names = ["train"]

        self._scaler = self._init_scaler(scaler_type=scaler_type,
                                         scaler_options=scaler_options,
                                         scaler=scaler,
                                         reader=reader)

        self._init_generators(generator_options=generator_options,
                              scaler=self._scaler,
                              reader=reader)

        self._init_model(model=model, model_options=model_options, compiler_options=compile_options)

    @abstractmethod
    def _init_model(self, model, model_options, compiler_options):
        ...

    @abstractmethod
    def _create_generator(self, reader: ProcessedSetReader, scaler: AbstractScaler,
                          **generator_options):
        ...

    def _init_scaler(self, scaler_type: str, scaler_options: dict, scaler: AbstractScaler,
                     reader: Union[ProcessedSetReader, SplitSetReader]):
        if scaler is not None:
            self._scaler = scaler
            return

        if isinstance(reader, SplitSetReader):
            reader = reader.train
        if not scaler_type in ["minmax", "standard"]:
            raise ValueError(
                f"Invalid scaler type: {scaler_type}. Must be either 'minmax' or 'standard'.")
        if scaler_type == "minmax":
            return MIMICMinMaxScaler(storage_path=self._storage_path, **scaler_options)
        elif scaler_type == "standard":
            return MIMICStandardScaler(storage_path=self._storage_path, **scaler_options)
        elif scaler_type == "maxabs":
            return MIMICMaxAbsScaler(storage_path=self._storage_path, **scaler_options)
        elif scaler_type == "robust":
            return MIMICRobustScaler(storage_path=self._storage_path, **scaler_options)

    @staticmethod
    def _check_generator_sanity(set_name: str, batch_size: int, reader: ProcessedSetReader,
                                generator: TFGenerator):
        if not generator.steps:
            if reader.subject_ids:
                raise ValueError(
                    f"{set_name.capitalize()} generator has no steps, while {len(reader.subject_ids)}"
                    f" subjects are present in reader. Consider reducing batch size: {batch_size}.")
            else:
                raise ValueError(
                    f"{set_name.capitalize()} reader has no subjects. Consider adjusting the {set_name} split size."
                )

    def _init_generators(self, generator_options: dict, scaler: AbstractScaler,
                         reader: Union[ProcessedSetReader, SplitSetReader]):
        if isinstance(reader, ProcessedSetReader):
            self._train_generator = self._create_generator(reader=reader,
                                                           scaler=scaler,
                                                           **generator_options)

            self._val_generator = None
            self._val_steps = None
        elif isinstance(reader, SplitSetReader):
            self._train_generator = self._create_generator(reader=reader.train,
                                                           scaler=scaler,
                                                           **generator_options)

            self._check_generator_sanity(set_name="train",
                                         batch_size=generator_options["batch_size"],
                                         reader=reader.train,
                                         generator=self._train_generator)

            if "val" in reader.split_names:
                self._split_names.append("val")
                self._val_generator = self._create_generator(reader=reader.val,
                                                             scaler=scaler,
                                                             **generator_options)

                self._val_steps = self._val_generator.steps
                self._check_generator_sanity(set_name="val",
                                             batch_size=generator_options["batch_size"],
                                             reader=reader.val,
                                             generator=self._val_generator)

            else:
                self._val_generator = None
                self._val_steps = None

            if "test" in reader.split_names:
                self._split_names.append("test")
                self._test_generator = self._create_generator(reader=reader.test,
                                                              scaler=scaler,
                                                              **generator_options)

                self._check_generator_sanity(set_name="test",
                                             batch_size=generator_options["batch_size"],
                                             reader=reader.test,
                                             generator=self._test_generator)

    @abstractmethod
    def fit(self,
            result_name: str,
            result_path: Path = None,
            epochs: int = None,
            patience: int = None,
            *args,
            **kwargs):
        ...
