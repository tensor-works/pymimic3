import pandas as pd
import datasets
from pathlib import Path
from preprocessing.scalers import MIMICStandardScaler
from managers import HistoryManager
from preprocessing.discretizers import Discretizer
from utils.IO import *
from pipelines import AbstractMIMICPipeline
from pathlib import Path
import datasets
import numpy as np
import pandas as pd

from multipledispatch import dispatch
from sklearn import metrics

from metrics import CustomCategoricalMSE
from utils.IO import *
from utils import make_prediction_vector
from preprocessing.preprocessors import MIMICPreprocessor
from models.callbacks import HistoryCheckpoint
from managers import CheckpointManager
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class MIMICPipeline(AbstractMIMICPipeline):
    """_summary_
    """

    def __init__(
            self,
            model,
            batch_size: int,
            compiler_config: dict,
            data_source_path: Path,
            data_storage_path: Path,
            epochs: int,
            model_name: str,
            output_type: str,
            task: str,
            split_config: dict = {},
            model_type: str = "neural_network",
            subset_size: int = None,
            callbacks: list = [],
            custom_objects: dict = {},
            patience: int = 10,  # TODO! default should be None
            root_path: str = "",
            save: bool = True):
        """_summary_

        Args:
            model (_type_):                                 compiled model object
            model_name (str): 
            window_length (int):                            window length in hours
            horizon (str):                                  can be hhourly, hourly, qdaily, hdaily, daily
            output_type (str):                              can be onehot, sparse, normal, None
            model_type (str, otpional):                     can be neural_network or regression
            batch_size (int):       
            epochs (int): 
            data_type (str, optional): .
            validation_fraction_split (float, optional):    Defaults to 0..
            test_fraction_split (float, optional):          Defaults to 0..
            split_method (str, optional):                   Can be date or time. Defaults to "time".
            n_bins (int, optional):                         Unbinned data if 0. Defaults to 0.
        """

        self.batch_size = batch_size
        self.callbacks = callbacks
        self.compiler_config = compiler_config
        self.custom_objects = custom_objects
        self.epochs = epochs
        self.model = model
        self.model_name = model_name
        self.output_type = output_type
        self.patience = patience
        self.split_config = split_config

        self.root_path = Path(root_path)
        self.data_storage_path = Path(data_storage_path)
        self.data_source_path = Path(data_source_path)

        self.save = save
        self.task = task
        self.subset_size = subset_size
        self.model_type = model_type

        self.hist_manager: HistoryManager = None
        self.preprocessor: MIMICPreprocessor = None
        self.inverse_scaling: bool = True

    def fit(self,
            timeseries: pd.DataFrame = None,
            episodic_data: pd.DataFrame = None,
            subject_diagnoses: pd.DataFrame = None,
            subject_icu_history: pd.DataFrame = None,
            reader: object = None):
        """_summary_

        Args:
            timeseries (pd.DataFrame): _description_
            episodic_data (pd.DataFrame): _description_
            subject_diagnoses (pd.DataFrame): _description_
            subject_icu_history (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        self._init_directories()
        self.hist_manager = HistoryManager(self.directories["model_path"])
        self.discretizer = Discretizer(config_dictionary=Path(self.directories["config_path"],
                                                              "discretizer_config.json"),
                                       eps=1e-6)

        normalizer_file = Path(self.directories["normalizer_path"],
                               f"normalizer_{self.task}_acorn-a.obj")

        self.normalizer = MIMICStandardScaler(normalizer_file)
        self.reader_flag = reader is not None  # affects init functions

        if self.reader_flag:
            self._iterative_processing(reader)

            readers, \
            ratios = datasets.train_test_split(source_path=self.directories["task_data"],
                                               model_path=self.directories["model_path"],
                                                **self.split_config)

            self._init_normalizer(readers["train"], preprocessor="discretizer")
            self._init_batch_readers(readers)
        else:
            self.generator_type = MIMICBatchGenerator
            X_subjects, y_subjects = self._processing(timeseries, episodic_data, subject_diagnoses,
                                                      subject_icu_history)
            info_io("Splitting the dataset.")
            X_dataset, \
            y_dataset = datasets.train_test_split(X_subjects,
                                                  y_subjects,
                                                  **self.split_config)

            self._init_normalizer(X_dataset["train"], preprocessor="discretizer")
            self.init_batch_generators(X_dataset, y_dataset)

        info_io(f"Starting model run for model: {self.model_name}!")

        self.train()

        return self.model, self.hist_manager.history

    def convert_custom_objects(self) -> None:
        """_summary_
        """

        if "categorical_mse" in self.compiler_config["metrics"]:
            cat_mse = CustomCategoricalMSE(self.preprocessor.bin_averages).average_categorical_mse
            self.compiler_config["metrics"].remove("categorical_mse")
            self.compiler_config["metrics"].append(cat_mse)
            self.custom_objects.update({"average_categorical_mse": cat_mse})

        if self.custom_objects:
            self.compiler_config = {
                key: (self.custom_objects[value] if value in list(self.custom_objects.keys()) else
                      value) for key, value in self.compiler_config.items()
            }
            self.compiler_config["metrics"] = [
                self.custom_objects[value] if value in list(self.custom_objects.keys()) else value
                for value in self.compiler_config["metrics"]
            ]

        return

    def train(self) -> None:
        """_summary_
        """
        self.model.summary()

        self.convert_custom_objects()

        manager = CheckpointManager(self.directories["model_path"],
                                    self.epochs,
                                    custom_objects=self.custom_objects)

        if not manager.is_empty():
            self.model = manager.load_model()

        self.model.compile(**self.compiler_config)

        es_callback = EarlyStopping(patience=self.patience, restore_best_weights=True)

        cp_callback = ModelCheckpoint(filepath=Path(self.directories["model_path"],
                                                    "cp-{epoch:04d}.ckpt"),
                                      save_weights_only=False,
                                      save_best_only=True,
                                      verbose=0)

        hist_callback = HistoryCheckpoint(self.history_file)

        # lr_decay = DecayLearningRate(4, 1)

        if self.history_file.is_file():
            if self.hist_manager.is_finished():
                info_io("Model has already been trained!")
                return

        if self.validation_fraction_split:
            validation_args = {
                "validation_data": self.generators["val"],
                "validation_steps": self.generators["val"].steps,
            }
        else:
            validation_args = {}

        info_io("Training model.")

        self.model.fit(self.generators["train"],
                       steps_per_epoch=self.generators["train"].steps,
                       epochs=self.epochs,
                       initial_epoch=manager.latest_epoch(),
                       callbacks=[cp_callback, hist_callback, es_callback] + self.callbacks,
                       **validation_args)

        self.hist_manager.finished()
        _, best_epoch = self.hist_manager.best

        manager.clean_directory(best_epoch)

        info_io("Training complete.")

    def _init_batch_reader_arguments(self):
        """_summary_
        """
        self.batch_reader_kwargs = {
            "discretizer": self.discretizer,
            "type": "cat",
            "normalizer": self.normalizer,
        }

        return

    def init_generator_arguments(self):
        """_summary_
        """
        self.generator_kwargs = {
            "discretizer": self.discretizer,
            "type": "cat",
            "normalizer": self.normalizer,
            "subset_size": self.subset_size
        }

        return

    def init_batch_generators(self, X_dataset: dict, y_dataset: dict):
        """_summary_

        Args:
            split_dataset (list): _description_

        Returns:
            _type_: _description_
        """
        self.init_generator_arguments()
        self.generators = {
            set_name:
                MIMICBatchGenerator(X_dataset[set_name], y_dataset[set_name],
                                    **self.generator_kwargs) for set_name in X_dataset.keys()
        }

        return self.generators

    def make_generator(self, X, y):
        """_summary_

        Args:
            X (list): _description_
        """
        self.init_generator_arguments()
        generator = MIMICBatchGenerator(X, y, **self.generator_kwargs)

        return generator

    def _batch_reader(self, reader, **kwargs):
        """_summary_

        Args:
            reader (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._init_batch_reader_arguments()
        self.batch_reader_kwargs.update(kwargs)
        return IterativeGenerator(reader, **self.batch_reader_kwargs)

    @dispatch(object)
    def predict_proba(self, generator, return_labels=False):
        """_summary_
        """

        if generator is None:
            generator = self.generators["test"]

        y_pred, y_true = make_prediction_vector(self.model,
                                                self.generator,
                                                batches=self.generator.steps,
                                                bin_averages=None)
        if self.output_type is None and self.inverse_scaling:
            y_pred = self.preprocessor.scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_true = self.preprocessor.scaler.inverse_transform(y_true.reshape(-1, 1))

        return y_pred, y_true

    @dispatch()
    def predict_proba(self, return_labels=False):
        """_summary_
        """
        return self.predict_proba(self.generators["test"])

    @dispatch(object)
    def predict(self, generator: object):
        """_summary_

        Args:
            generator (object): _description_

        Returns:
            _type_: _description_
        """

        if generator is None:
            generator = self.generators["test"]

        y_pred, y_true = make_prediction_vector(self.model,
                                                self.generator,
                                                batches=self.generator.steps,
                                                bin_averages=None)
        if self.output_type is None and self.inverse_scaling:
            y_pred = self.preprocessor.scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_true = self.preprocessor.scaler.inverse_transform(y_true.reshape(-1, 1))

        return y_pred, y_true

    @dispatch()
    def predict(self):
        """_summary_
        """
        return self.predict(self.generators["test"])
