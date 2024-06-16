import datasets
import re
from pathlib import Path
from models.callbacks import HistoryCheckpoint
from generators.tf2 import TFGenerator
from preprocessing.scalers import AbstractScaler
from managers import HistoryManager
from managers import CheckpointManager
from datasets.readers import ProcessedSetReader
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.IO import *
from pipelines import AbstractPipeline


class TFPipeline(AbstractPipeline):

    def _create_generator(self, reader: ProcessedSetReader, scaler: AbstractScaler,
                          **generator_options):
        return TFGenerator(reader=reader, scaler=scaler, **generator_options)

    def _init_callbacks(self,
                        patience: int = None,
                        restore_best_weights=True,
                        save_weights_only: bool = False,
                        save_best_only: bool = True):
        self._standard_callbacks = []
        if "val" in self._split_names:
            if patience is not None:
                es_callback = EarlyStopping(patience=patience,
                                            restore_best_weights=restore_best_weights)
                self._standard_callbacks.append(es_callback)

        if self._result_path is not None:
            cp_pattern = str(Path(self._result_path, "cp-{epoch:04d}.ckpt"))
            cp_callback = ModelCheckpoint(filepath=cp_pattern,
                                          save_weights_only=save_weights_only,
                                          save_best_only=save_best_only,
                                          verbose=0)
            self._standard_callbacks.append(cp_callback)

            hist_callback = HistoryCheckpoint(self._result_path)
            self._standard_callbacks.append(hist_callback)

    def _init_managers(self, epochs: int):
        self._hist_manager = HistoryManager(str(self._result_path))

        self._manager = CheckpointManager(str(self._result_path), epochs, custom_objects={})

    def fit(self,
            epochs: int,
            no_subdirs: bool = False,
            result_name: str = None,
            patience: int = None,
            restore_best_weights=True,
            save_weights_only: bool = False,
            class_weight: dict = None,
            sample_weight: dict = None,
            save_best_only: bool = True,
            callbacks: list = [],
            validation_freq: int = 1,
            restore_last_run: bool = False):

        self._init_result_path(result_name=result_name,
                               restore_last_run=restore_last_run,
                               no_subdirs=no_subdirs)
        info_io(f"Training model in directory\n{self._result_path}")
        self._init_managers(epochs)
        self._init_callbacks(patience=patience,
                             restore_best_weights=restore_best_weights,
                             save_weights_only=save_weights_only,
                             save_best_only=save_best_only)

        self._model.fit(self._train_generator,
                        validation_data=self._val_generator,
                        epochs=epochs,
                        steps_per_epoch=self._train_generator.steps,
                        callbacks=callbacks + self._standard_callbacks,
                        class_weight=class_weight,
                        sample_weight=sample_weight,
                        initial_epoch=self._manager.latest_epoch(),
                        validation_steps=self._val_steps,
                        validation_freq=validation_freq)

        self._hist_manager.finished()
        _, best_epoch = self._hist_manager.best
        self._manager.clean_directory(best_epoch)

        return self._model


if __name__ == "__main__":
    from models.tf2.lstm import LSTMNetwork
    from tests.tsettings import *
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task="IHM")

    reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)

    model = LSTMNetwork(10,
                        0.2,
                        59,
                        bidirectional=False,
                        recurrent_dropout=0.,
                        task=None,
                        target_repl=False,
                        output_dim=1,
                        depth=1)

    pipe = TFPipeline(storage_path=Path(TEMP_DIR, "tf_pipeline"),
                      reader=reader,
                      model=model,
                      compile_options={
                          "optimizer": "adam",
                          "loss": "binary_crossentropy"
                      },
                      generator_options={
                          "batch_size": 1,
                          "shuffle": True
                      }).fit(epochs=10, save_best_only=False)

    pipe = TFPipeline(storage_path=Path(TEMP_DIR, "tf_pipeline"),
                      reader=reader,
                      model=model,
                      compile_options={
                          "optimizer": "adam",
                          "loss": "binary_crossentropy"
                      },
                      generator_options={
                          "batch_size": 1,
                          "shuffle": True
                      }).fit(epochs=10, save_best_only=False)

    pipe = TFPipeline(storage_path=Path(TEMP_DIR, "tf_pipeline"),
                      reader=reader,
                      model=model,
                      compile_options={
                          "optimizer": "adam",
                          "loss": "binary_crossentropy"
                      },
                      generator_options={
                          "batch_size": 1,
                          "shuffle": True
                      }).fit(epochs=20, save_best_only=False, restore_last_run=True)
