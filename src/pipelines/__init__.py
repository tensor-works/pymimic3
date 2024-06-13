import re
import datasets
from typing import Union
from pathlib import Path
from generators.tf2 import TFGenerator
from preprocessing.scalers import AbstractScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from datasets.readers import ProcessedSetReader
from datasets.readers import ProcessedSetReader, SplitSetReader
from abc import ABC, abstractmethod
from utils.IO import *


class AbstractPipeline(ABC):
    """
    Abstract base class for pipelines used in data processing and model training.

    Parameters
    ----------
    storage_path : Path
        Path to the directory where results and intermediate files will be stored.
    reader : ProcessedSetReader or SplitSetReader
        Reader object to read the processed data.
    model : object
        The model to be trained.
    generator_options : dict, optional
        Options for the data generator.
    model_options : dict, optional
        Options for the model.
    scaler_options : dict, optional
        Options for the scaler.
    compile_options : dict, optional
        Options for model compilation.
    split_options : dict, optional
        Options for data splitting.
    scaler : AbstractScaler, optional
        Scaler object to scale the data.
    scaler_type : str, optional
        Type of scaler to use ('minmax', 'standard', 'maxabs', 'robust').

    Attributes
    ----------
    _storage_path : Path
        Path to the directory where results and intermediate files will be stored.
    _model : object
        The model to be trained.
    _generator_options : dict
        Options for the data generator.
    _data_split_options : dict
        Options for data splitting.
    _split_names : list
        Names of data splits.
    _reader : Union[ProcessedSetReader, SplitSetReader]
        Reader object to read the processed data.
    _scaler : AbstractScaler
        Scaler object to scale the data.
    _train_generator : TFGenerator
        Data generator for the training set.
    _val_generator : TFGenerator
        Data generator for the validation set.
    _val_steps : int
        Number of steps in the validation set.
    _test_generator : TFGenerator
        Data generator for the test set.
    _result_path : Path
        Path to the directory where results will be stored.
    """

    def __init__(
        self,
        storage_path: Path,
        reader: Union[ProcessedSetReader, SplitSetReader],
        model,
        generator_options: dict = {},
        model_options: dict = {},
        scaler_options: dict = {},
        compile_options: dict = {},
        split_options: dict = {},
        scaler: AbstractScaler = None,
        scaler_type: str = "minmax",
    ):
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._generator_options = generator_options
        self._data_split_options = split_options
        self._split_names = ["train"]
        self._reader = self._split_data(data_split_options=split_options, reader=reader)

        self._scaler = self._init_scaler(storage_path=storage_path,
                                         scaler_type=scaler_type,
                                         scaler_options=scaler_options,
                                         scaler=scaler,
                                         reader=reader)

        self._init_generators(generator_options=generator_options,
                              scaler=self._scaler,
                              reader=reader)

        self._init_model(model=model, model_options=model_options, compiler_options=compile_options)

    def _init_model(self, model, model_options, compiler_options):
        """
        Initializes the model.
        """
        if isinstance(model, type):
            self._model = model(**model_options)
        if hasattr(model, "optimizer") and model.optimizer is None:
            self._model.compile(**compiler_options)

    def _split_data(self, data_split_options: dict, reader: Union[ProcessedSetReader,
                                                                  SplitSetReader]):
        """
        Splits the data according to the provided options.
        """
        if isinstance(reader, ProcessedSetReader) and data_split_options:
            return datasets.train_test_split(reader, **data_split_options)
        return reader

    @abstractmethod
    def _create_generator(self, reader: ProcessedSetReader, scaler: AbstractScaler,
                          **generator_options):
        """
        Creates a data generator.
        """
        ...

    def _init_scaler(self, storage_path: Path, scaler_type: str, scaler_options: dict,
                     scaler: Union[AbstractScaler, type], reader: Union[ProcessedSetReader,
                                                                        SplitSetReader]):
        """
        Initializes the scaler.
        """
        if isinstance(reader, SplitSetReader):
            reader = reader.train
        if not scaler_type in ["minmax", "standard", "maxabs", "robust"]:
            raise ValueError(
                f"Invalid scaler type: {scaler_type}. Must be either 'minmax' or 'standard'.")

        if scaler is not None and isinstance(scaler, AbstractScaler):
            return scaler.fit_reader(reader)
        elif scaler is not None and isinstance(scaler, type):
            scaler = scaler(storage_path=storage_path, **scaler_options)
            return scaler.fit_reader(reader)
        elif scaler_type == "minmax":
            scaler = MinMaxScaler(storage_path=storage_path, **scaler_options)
            return scaler.fit_reader(reader)
        elif scaler_type == "standard":
            scaler = StandardScaler(storage_path=storage_path, **scaler_options)
            return scaler.fit_reader(reader)
        elif scaler_type == "maxabs":
            scaler = MaxAbsScaler(storage_path=storage_path, **scaler_options)
            return scaler.fit_reader(reader)
        elif scaler_type == "robust":
            scaler = RobustScaler(storage_path=storage_path, **scaler_options)
            return scaler.fit_reader(reader)

    @staticmethod
    def _check_generator_sanity(set_name: str, reader: ProcessedSetReader, generator: TFGenerator,
                                generator_options: dict):
        """
        Checks wether the generator is empty.
        """
        if not len(generator):
            if reader.subject_ids:
                msg = f"{set_name.capitalize()} generator has no steps, while {len(reader.subject_ids)}"
                msg += f" subjects are present in reader. "
                if "batch_size" in generator_options:
                    msg += f"Consider reducing batch size: {generator_options['batch_size']}."

                raise ValueError(msg)
            else:
                raise ValueError(
                    f"{set_name.capitalize()} reader has no subjects. Consider adjusting the {set_name} split size."
                )

    def _init_generators(self, generator_options: dict, scaler: AbstractScaler,
                         reader: Union[ProcessedSetReader, SplitSetReader]):
        """
        Initializes the test, val and train data generators if test, val and train are part of
        the split.
        """
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
                                         generator_options=generator_options,
                                         reader=reader.train,
                                         generator=self._train_generator)

            if "val" in reader.split_names:
                self._split_names.append("val")
                self._val_generator = self._create_generator(reader=reader.val,
                                                             scaler=scaler,
                                                             **generator_options)

                self._val_steps = len(self._val_generator)
                self._check_generator_sanity(set_name="val",
                                             generator_options=generator_options,
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
                                             generator_options=generator_options,
                                             reader=reader.test,
                                             generator=self._test_generator)

    def _init_result_path(self,
                          result_name: str,
                          restore_last_run: bool = False,
                          no_subdirs: bool = False):
        """
        Initializes the result path. If restore last run is set to True, the result path will be the
        the previous numerical result path. If no_subdirs is set to True, the result path will be the
        the storage path. If a result name is provided, the result path will be the storage path with
        the result name.
        """
        if no_subdirs:
            if result_name is not None:
                warn_io("Ignoring result_name, as no_subdirs is set to True.")
            self._result_path = self._storage_path
        elif result_name is not None:
            self._result_path = Path(self._storage_path, result_name)
        else:
            # Iterate over files in the directory
            pattern = re.compile(r"(\d+)")
            result_numbers = []
            for file in self._storage_path.iterdir():
                if file.is_dir() and file.name.startswith("results"):
                    match = pattern.search(file.name)
                    if match:
                        result_numbers.append(int(match.group(0)))

            # Determine the next number
            if not result_numbers:
                next_number = 0
            else:
                next_number = max(result_numbers, default=0) + int(not restore_last_run)
            self._result_path = Path(self._storage_path, f"results{next_number:04d}")
        self._result_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fit(self, epochs: int, result_name: str = None, no_subdirs: bool = False, *args, **kwargs):
        """
        Fits the model to the data.
        """
        ...

    # def test(self):
    #     self._model.test(self._test_generator)
