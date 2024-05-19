import datasets
from typing import Union
from pathlib import Path
from generators.scikit import ScikitGenerator
from preprocessing.scalers import AbstractScaler
from preprocessing.imputers import AbstractImputer, PartialImputer
from utils.IO import *
from datasets.readers import ProcessedSetReader, SplitSetReader
from pipelines import AbstractPipeline


class TorchPipeline(AbstractPipeline):

    def __init__(self,
                 storage_path: Path,
                 reader: Union[ProcessedSetReader, SplitSetReader],
                 model,
                 generator_options: dict,
                 model_options: dict = {},
                 scaler_options: dict = {},
                 compile_options: dict = {},
                 split_options: dict = {},
                 imputer_options: dict = {},
                 scaler: AbstractScaler = None,
                 scaler_type: str = "minmax",
                 imputer: AbstractImputer = None):

        self._imputer = self._init_imputer(storage_path=storage_path,
                                           imputer=imputer,
                                           **imputer_options)

        scaler_options["imputer"] = self._imputer

        super().__init__(storage_path=storage_path,
                         reader=reader,
                         model=model,
                         generator_options=generator_options,
                         model_options=model_options,
                         scaler_options=scaler_options,
                         compile_options=compile_options,
                         split_options=split_options,
                         scaler=scaler,
                         scaler_type=scaler_type)

    def _init_imputer(self, storage_path: Path, imputer: AbstractImputer, imputer_options):
        if imputer is not None and isinstance(imputer, AbstractImputer):
            return imputer.fit_reader(reader)
        elif imputer is not None and isinstance(imputer, type):
            imputer = imputer(storage_path=storage_path, **imputer_options)
            return imputer.fit_reader(reader)
        elif imputer is None:
            imputer = PartialImputer(storage_path=storage_path, **imputer_options)
            return imputer.fit_reader(reader)

    def _create_generator(self, reader: ProcessedSetReader, scaler: AbstractScaler,
                          **generator_options):
        return ScikitGenerator(reader=reader, scaler=scaler, **generator_options)

    def fit(self,
            epochs: int,
            result_name: str = None,
            patience: int = None,
            save_best_only: bool = True,
            restore_best_weights: bool = True,
            sample_weights: dict = None,
            val_frequency=1,
            restore_last_run: bool = False,
            **kwargs):

        self._init_result_path(result_name, restore_last_run)
        info_io(f"Training model in directory\n{self._result_path}")
        self._model.fit(self._train_generator,
                        model_path=self._result_path,
                        epochs=epochs,
                        patience=patience,
                        save_best_only=save_best_only,
                        restore_best_weights=restore_best_weights,
                        sample_weights=sample_weights,
                        val_frequency=val_frequency,
                        val_generator=self._val_generator,
                        **kwargs)

        return self._model


if __name__ == "__main__":
    from tests.settings import *
    for task_name in TASK_NAMES:
        reader = datasets.load_data(chunksize=75836,
                                    source_path=TEST_DATA_DEMO,
                                    storage_path=SEMITEMP_DIR,
                                    engineer=True,
                                    task=task_name)
