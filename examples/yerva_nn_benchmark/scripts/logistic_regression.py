import datasets
from river import multiclass
from pathlib import Path
from datasets.readers import SplitSetReader
from models.stream.linear_model import LogisticRegression, LinearRegression, MultiOutputLogisticRegression
from pipelines.stream import RiverPipeline


def run_log_reg(task_name: str, reader: SplitSetReader, storage_path: Path, metrics: list,
                params: dict):
    if task_name == "LOS":
        run_linear_reg(reader=reader, storage_path=storage_path, metrics=metrics, params=params)
    elif task_name in ["DECOMP", "IHM"]:
        run_binary_log_reg(reader=reader, storage_path=storage_path, metrics=metrics, params=params)
    elif task_name == "PHENO":
        run_multioutput_log_reg(reader=reader,
                                storage_path=storage_path,
                                metrics=metrics,
                                params=params)


def run_binary_log_reg(reader: SplitSetReader, storage_path: Path, metrics: list, params: dict):
    model = LogisticRegression(metrics=metrics, **params)
    pipe = RiverPipeline(storage_path=storage_path, reader=reader, model=model)
    pipe.fit(no_subdirs=True)
    pipe.test()


def run_multioutput_log_reg(reader: SplitSetReader, storage_path: Path, metrics: list,
                            params: dict):
    model = MultiOutputLogisticRegression(metrics=metrics, **params)
    pipe = RiverPipeline(storage_path=storage_path, reader=reader, model=model)
    pipe.fit(no_subdirs=True)
    pipe.test()


def run_linear_reg(reader: SplitSetReader, storage_path: Path, metrics: list, params: dict):
    model = LinearRegression(metrics=metrics, **params)
    pipe = RiverPipeline(storage_path=storage_path, reader=reader, model=model)
    pipe.fit(no_subdirs=True)
    pipe.test()
