import sys
import numpy as np
from pathlib import Path
from pathlib import Path
import datasets
import numpy as np
import pandas as pd

from multipledispatch import dispatch
from utils.IO import *
from utils import make_prediction_vector
from models.callbacks import HistoryCheckpoint
from managers import ReducedCheckpointManager
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from managers import HistoryManager
from preprocessing.discretizers import Discretizer
from preprocessing.scalers import MIMICStandardScaler
from preprocessing.imputers import BatchImputer
from utils.IO import *
from pipelines import AbstractMIMICPipeline
from generators.regression import MIMICBatchReader
from tensorflow.keras.metrics import AUC
from utils import update_json
# from netcal.scaling import TemperatureScaling, BetaCalibration, LogisticCalibration
# from netcal.binning import HistogramBinning, IsotonicRegression
# from netcal.metrics import ECE
from sklearn.calibration import calibration_curve, CalibrationDisplay
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score
from functools import partial
from metrics import AbstractMIMICMetric
# from metrics import auc_roc_macro, auc_roc_micro

import warnings

# warnings.filterwarnings("error")


class MIMICPipeline(AbstractMIMICPipeline):
    """_summary_
    """

    def __init__(self,
                 model,
                 metrics: dict,
                 model_name: str,
                 root_path: Path,
                 task: str,
                 framework: str,
                 output_type: str = "sparse",
                 split_config: dict = {},
                 compiler_config: dict = {},
                 patience: int = None,
                 callbacks: list = [],
                 custom_objects: list = [],
                 save: bool = True) -> None:
        """_summary_

        Args:
            custom_objects (dict): _description_
            data_source_path (Path): _description_
            model (object): _description_
            model_name (str): _description_
            output_type (str): _description_
            root_path (Path): _description_
            test_fraction_split (float, optional): _description_. Defaults to 0.
        """
        generation_switch = {"PHENO": False, "IHM": False, "LOS": True, "DECOMP": True}
        # TODO! setting

        self.metrics = metrics
        self.model = model
        self.model_name = model_name
        self.output_type = output_type
        self.root_path = Path(root_path)
        self.task = task
        self.framework = framework
        self.save = save
        self.generator_flag = generation_switch[task] or framework == "tf2"
        self.patience = patience
        self.callbacks = callbacks
        self.custom_objects = custom_objects
        self.compiler_config = compiler_config
        self.metrics = metrics
        self.split_config = split_config

    def _init_batch_reader_arguments(self):
        """_summary_
        """
        self.batch_reader_kwargs = {
            "discretizer": self.imputer,
            "type": "cat",
            "normalizer": self.normalizer,
        }

        return

    def _init_split_settings(self, data_path: Path):
        """_summary_
        """
        if "attribute" in self.split_config:
            split_info_df = Path(Path(data_path).parent, "extracted", "subject_info_df.csv")
            self.split_config.update({"split_info_path": split_info_df})
            self.split_config.update({"progress_file_path": Path()})
        elif not (self.reader_flag or self.generator_flag):
            self.split_config.update({"method": "sample"})

    def _train(self, X_dataset, y_dataset) -> None:
        """_summary_
        """
        X_train = X_dataset["train"]
        X_train = np.array(self.imputer.transform(X_train), dtype=np.float32)

        self.normalizer.fit(X_train)

        X_train = self.normalizer.transform(X_train)
        y_train = np.squeeze(y_dataset["train"])

        info_io(f"Fit the model: {self._model_name}")
        self.model.fit(X_train, y_train)

        self.model.evaluate(X_train, y_train, self.metrics, self.directories["model_path"], "train")

        def evaluate_subset(set_name):
            X_test = np.array(self.imputer.transform(X_dataset[set_name]), dtype=np.float32)
            X_test = self.normalizer.transform(X_test)
            y_test = np.squeeze(y_dataset[set_name])
            self.model.evaluate(X_test, y_test, self.metrics, self.directories["model_path"],
                                "train")

        for set_name in ["test", "val"]:
            if set_name in X_dataset:
                evaluate_subset(set_name)
        return

    def _iterative_train(self) -> None:
        """_summary_
        """
        info_io(f"Starting model run for case {self.model_name} and task {self.task}!")
        epochs = self.model.max_iter
        optional_args = dict()
        if "val" in self.generators:
            optional_args.update({
                "validation_data": self.generators["val"],
                "validation_steps": self.generators["val"].steps,
            })

        if self.framework == "tf2":
            callbacks = list()
            if self.patience:
                callbacks.append(EarlyStopping(patience=self.patience, restore_best_weights=True))
            if self.save:
                callbacks.append(
                    ModelCheckpoint(filepath=Path(self.directories["model_path"],
                                                  "cp-{epoch:04d}.ckpt"),
                                    save_weights_only=False,
                                    save_best_only=True))
                callbacks.append(HistoryCheckpoint(self.history_file))
                callbacks.append(
                    AbstractMIMICMetric(self.generators["train"], self.generators["val"],
                                        self.model, self.metrics))

            self.model.load(self.directories["model_path"])
            manager = ReducedCheckpointManager(self.directories["model_path"])

            if self.history_file.is_file():
                if self.hist_manager.is_finished():
                    info_io("Model has already been trained!")
                    # return

            optional_args.update({
                "callbacks": callbacks + self.callbacks,
                "initial_epoch": self.model.latest_epoch(self.directories["model_path"])
            })

        self.model.compile(**self.compiler_config)
        info_io("Training model.")

        if self.framework == "tf2":
            self.model.summary()

        import logging

        class CustomHandler(logging.Handler):

            def emit(self, record):
                log_entry = self.format(record)

        class ColoredErrorStream:

            def __init__(self, stream):
                self.stream = stream

            def write(self, message):
                # Check for TensorFlow internal log messages
                if "tensorflow" in message:
                    # Apply yellow color
                    colored_message = f"\033[93m{message}\033[0m"
                else:
                    colored_message = message
                self.stream.write(colored_message)

            def flush(self):
                self.stream.flush()

            def fileno(self):
                return self.stream.fileno()

        sys.stderr = ColoredErrorStream(sys.stderr)
        import tensorflow as tf
        tf.get_logger()

        class ColoredOutputStream:

            def __init__(self, stream):
                self.stream = stream

            def write(self, message):
                # Check if the message meets your criteria for coloring
                # For example, let's color all messages that contain 'Epoch'
                if 'Epoch' in message:
                    # Apply yellow color
                    colored_message = f"\033[93m{message}\033[0m"
                    self.stream.write(colored_message)
                else:
                    self.stream.write(message)

            def flush(self):
                self.stream.flush()

            def fileno(self):
                return self.stream.fileno()

        sys.stdout = ColoredOutputStream(sys.stdout)

        # Add custom handler to TensorFlow's logger
        # logger = logging.getLogger('tensorflow')
        # logger.setLevel(logging.INFO)
        # logger.addHandler(CustomHandler())

        class CustomFormatter(logging.Formatter):
            """Custom formatter to add color to log messages."""

            def format(self, record):
                if record.levelno == logging.WARNING:
                    # Yellow color for warnings
                    record.msg = f"\033[93m{record.msg}\033[0m"
                elif record.levelno == logging.ERROR:
                    # Red color for errors
                    record.msg = f"\033[91m{record.msg}\033[0m"
                return super().format(record)

        # Get TensorFlow logger
        logger = tf.get_logger()

        # Set custom formatter
        formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.handlers = [handler]

        for h in logger.handlers:
            h.setFormatter(formatter)
        # tf.logging.info("testing")
        # tf.logging.debug("debug mesg")
        # tf.logging.warning("a warning mesg")

        self.model.fit(self.generators["train"],
                       steps_per_epoch=self.generators["train"].steps,
                       epochs=epochs,
                       **optional_args)

        if self.framework == "tf2":
            self.hist_manager.finished()
            _, best_epoch = self.hist_manager.best

            manager.clean_directory(best_epoch)

        y_true = list()
        y_pred = list()
        for _ in range(self.generators["train"].steps):
            X, y = self.generators["train"].next()
            y_true.append(y)
            y_pred.append(self.model.predict(X))

        info_io("Training complete.")

    def fit(self,
            X_subjects: pd.DataFrame = None,
            y_subjects: pd.DataFrame = None,
            data_path: Path = None):
        """_summary_

        Args:
            timeseries (pd.DataFrame): _description_
            episodic_data (pd.DataFrame): _description_
            subject_diagnoses (pd.DataFrame): _description_
            subject_icu_history (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        self.reader_flag = data_path is not None  # affects init functions
        self._data_path = data_path
        self._init_directories()
        self._init_split_settings(data_path)
        self.hist_manager = HistoryManager(self.directories["model_path"])
        self.discretizer = Discretizer(config_dictionary=Path(self.directories["config_path"],
                                                              "discretizer_config.json"),
                                       eps=1e-6)

        normalizer_file = Path(self.directories["model_path"], "objects",
                               f"normalizer_{self.task}_acorn-a.obj")
        normalizer_file.parent.mkdir(parents=True, exist_ok=True)
        imputer_file = Path(self.directories["model_path"], "objects",
                            f"imputer_{self.task}_acorn-a.obj")
        self.normalizer = MIMICStandardScaler(normalizer_file)
        self.imputer = BatchImputer(missing_values=np.nan,
                                    strategy='mean',
                                    verbose=0,
                                    copy=True,
                                    storage_path=imputer_file)

        if self.reader_flag:
            if self.generator_flag:
                readers, \
                ratios = datasets.train_test_split(
                    source_path=self._data_path,
                    model_path=self.directories["model_path"],
                    **self.split_config)

                self._init_imputer(readers["train"])
                self._init_normalizer(readers["train"], preprocessor="imputer")
                self._init_batch_readers(readers)
                self._iterative_train()
            else:
                X, y = self.read_dataset()
                X_dataset, y_dataset = datasets.train_test_split(X, y, **self.split_config)

                self._init_imputer(X_dataset["train"])
                self._init_normalizer(X_dataset["train"], preprocessor="imputer")
                self._train(X_dataset, y_dataset)
        else:
            info_io("Splitting the dataset.")
            X_dataset, \
            y_dataset = datasets.train_test_split(X_subjects,
                                                  y_subjects,
                                                  **self.split_config)

            self._init_imputer(X_dataset["train"])
            self._init_normalizer(X_dataset["train"], preprocessor="imputer")

            self._train(X_dataset, y_dataset)

        return self.model, self.hist_manager.history

    def _batch_reader(self, reader, **kwargs):
        """_summary_

        Args:
            reader (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._init_batch_reader_arguments()
        self.batch_reader_kwargs.update(kwargs)
        return MIMICBatchReader(reader, **self.batch_reader_kwargs)

    def read_dataset(self, stay_identifiers=False):
        info_io("Splitting the dataset.")
        X = pd.read_csv(Path(self._data_path, "X.csv"))
        y = pd.read_csv(Path(self._data_path, "y.csv"))
        identifier_names = ["subject_id", "stay_id"]
        if not stay_identifiers \
           and any(X.columns.isin(identifier_names)) \
           and any(y.columns.isin(identifier_names)):
            X = X.drop(columns=identifier_names)
            y = y.drop(columns=identifier_names)
        X = X.to_numpy()
        y = y.to_numpy()

        return X, y

    @dispatch(object)
    def predict(self, generator: object = None):
        """_summary_

        Args:
            generator (object): _description_

        Returns:
            _type_: _description_
        """

        if generator is None:
            generator = self.generators["test"]

        y_pred, y_true = make_prediction_vector(self.model,
                                                generator,
                                                batches=generator.steps,
                                                bin_averages=None)
        if self.output_type is None and self.inverse_scaling:
            y_pred = self.preprocessor.scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_true = self.preprocessor.scaler.inverse_transform(y_true.reshape(-1, 1))

        return y_pred, y_true

    @dispatch(object)
    def predict_proba(self, generator: object = None, return_labels: bool = False):
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
