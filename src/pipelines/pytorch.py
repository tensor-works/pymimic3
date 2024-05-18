from pathlib import Path
import datasets
from generators.pytorch import TorchGenerator
from preprocessing.scalers import AbstractScaler, MIMICMinMaxScaler, MIMICStandardScaler, MIMICMaxAbsScaler, MIMICRobustScaler
from utils.IO import *
from datasets.readers import ProcessedSetReader, SplitSetReader
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pipelines import AbstractPipeline


class TorchPipeline(AbstractPipeline):

    def _create_generator(self, reader: ProcessedSetReader, scaler: AbstractScaler,
                          **generator_options):
        return TorchGenerator(reader=reader, scaler=scaler, **generator_options)

    def _init_model(self, model, model_options, compiler_options):
        if isinstance(model, type):
            self._model = model(**model_options)

        if model.optimizer is None:
            self._model.compile(**compiler_options)

    def fit(self,
            result_name: str,
            epochs: int,
            result_path: Path = None,
            patience: int = None,
            save_best_only: bool = True,
            restore_best_weights: bool = True,
            sample_weights: dict = None,
            val_frequency=1):
        self._model.fit(self,
                        self._train_generator,
                        epochs=epochs,
                        patience=patience,
                        save_best_only=save_best_only,
                        restore_best_weights=restore_best_weights,
                        sample_weights=sample_weights,
                        val_frequency=val_frequency,
                        val_generator=self._val_generator)

        return self._model


if __name__ == "__main__":
    ...
