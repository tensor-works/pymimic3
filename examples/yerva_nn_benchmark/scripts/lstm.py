from models.pytorch.lstm import LSTMNetwork
from tests.settings import *
from pipelines.pytorch import TorchPipeline
from pathlib import Path
from datasets.readers import SplitSetReader
from copy import deepcopy


def run_standard_lstm(task_name: str, reader: SplitSetReader, storage_path: Path, metrics: list,
                      params: dict):

    params = deepcopy(params)

    model = LSTMNetwork(**params.pop("model"))
    training_params = params.pop("training")
    params["compile_options"].update({"metrics": metrics})
    pipe = TorchPipeline(storage_path=storage_path, reader=reader, model=model,
                         **params).fit(no_subdirs=True, **training_params)


def run_binary_lstm(reader: SplitSetReader, storage_path: Path, metrics: list, params: dict):
    params = deepcopy(params)
    model = LSTMNetwork(**params.pop("model"))
    training_params = params.pop("training")
    if "compile_options" in params:
        params["compile_options"].update(params.pop("compile_options"))
    else:
        params["compile_options"] = params.pop("compile_options")
    pipe = TorchPipeline(storage_path=storage_path, reader=reader, model=model,
                         **params).fit(no_subdirs=True, **training_params)


def run_multilabel_lstm(reader: SplitSetReader, storage_path: Path, metrics: list, params: dict):
    ...


def run_regression_lstm(reader: SplitSetReader, storage_path: Path, metrics: list, params: dict):
    ...
