import datasets
from typing import Union
from pathlib import Path
from generators.stream import RiverGenerator
from preprocessing.scalers import AbstractScaler
from preprocessing.imputers import AbstractImputer, PartialImputer
from datasets.readers import ProcessedSetReader, SplitSetReader
from pipelines import AbstractPipeline
from utils.IO import *
from settings import *


class RiverPipeline(AbstractPipeline):

    def __init__(self,
                 storage_path: Path,
                 reader: Union[ProcessedSetReader, SplitSetReader],
                 model,
                 generator_options: dict = {},
                 model_options: dict = {},
                 scaler_options: dict = {},
                 split_options: dict = {},
                 imputer_options: dict = {},
                 scaler: AbstractScaler = None,
                 scaler_type: str = "minmax",
                 imputer: AbstractImputer = None):

        self._reader = self._split_data(data_split_options=split_options, reader=reader)
        self._imputer = self._init_imputer(storage_path=storage_path,
                                           imputer=imputer,
                                           imputer_options=imputer_options,
                                           reader=reader)

        scaler_options["imputer"] = self._imputer

        super().__init__(storage_path=storage_path,
                         reader=reader,
                         model=model,
                         generator_options=generator_options,
                         model_options=model_options,
                         scaler_options=scaler_options,
                         compile_options={},
                         split_options=split_options,
                         scaler=scaler,
                         scaler_type=scaler_type)

    def _init_imputer(self, storage_path: Path, imputer_options: dict, imputer: AbstractImputer,
                      reader: Union[ProcessedSetReader, SplitSetReader]):
        if isinstance(reader, SplitSetReader):
            reader = reader.train
        if imputer is not None and isinstance(imputer, AbstractImputer):
            return imputer.fit_reader(reader)
        elif imputer is not None and imputer(imputer, type):
            imputer = imputer(storage_path=storage_path, **imputer_options)
            return imputer.fit_reader(reader)
        elif imputer is None:
            imputer = PartialImputer(storage_path=storage_path, **imputer_options)
            return imputer.fit_reader(reader)

    def _create_generator(self, reader: ProcessedSetReader, scaler: AbstractScaler,
                          **generator_options):
        return RiverGenerator(reader=reader, scaler=scaler, **generator_options)

    def fit(self,
            result_name: str = None,
            no_subdirs: bool = False,
            restore_last_run: bool = False,
            **kwargs):

        self._init_result_path(result_name=result_name,
                               restore_last_run=restore_last_run,
                               no_subdirs=no_subdirs)

        info_io(f"Training model in directory\n{self._result_path}")
        self._model.fit(self._train_generator,
                        model_path=self._result_path,
                        val_generator=self._val_generator,
                        **kwargs)

        return self._model

    def _allowed_key(self, key: str):
        return not any([key.endswith(metric) for metric in TEXT_METRICS])

    def print_metrics(self, metrics: dict):
        stringified_dict = {k: str(v) for k, v in metrics.items()}

        # Find the maximum length of keys and values
        max_key_length = max(len(key) for key in stringified_dict.keys()) + 4
        max_value_length = max(
            len(value) for key, value in stringified_dict.items() if self._allowed_key(key)) + 4

        # Print the header with calculated padding
        msg = ""
        msg += f"{'Key':<{max_key_length}} {'Value':<{max_value_length}}\n"
        msg += '-' * (max_key_length + max_value_length + 1) + "\n"

        # Print the dictionary items with calculated padding
        non_numeric_values = None
        for key, value in stringified_dict.items():
            if self._allowed_key(key):
                msg += f"{key:<{max_key_length}} {value:<{max_value_length}}" + "\n"
            else:
                non_numeric_values = (key, value)
        if non_numeric_values is not None:
            msg += "\n" + "\n".join(non_numeric_values)

        info_io(msg)

    def test(self, **kwargs):

        info_io(f"Testing model in directory\n{self._result_path}")
        metrics = self._model.test(self._test_generator, **kwargs)
        self.print_metrics(metrics)

        return metrics


if __name__ == '__main__':
    import datasets
    from tests.tsettings import *
    # from models.stream.linear_model import LogisticRegression
    from metrics.stream import PRAUC

    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                engineer=True,
                                task="IHM")

    model_path = Path(TEMP_DIR, "arf_ihm")
    reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)
    # model = LogisticRegression(model_path=model_path, metrics=["accuracy", PRAUC])
    pipe = RiverPipeline(storage_path=Path(TEMP_DIR, "river_pipeline"),
                         reader=reader,
                         model=model,
                         generator_options={
                             "shuffle": True
                         }).fit()
