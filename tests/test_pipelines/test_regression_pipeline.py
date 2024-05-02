'''
import datasets
import os
import pandas as pd
import json
import shutil
from pathlib import Path
from utils.IO import *
from settings import *
from datasets.readers import ProcessedSetReader
import numpy as np
from pipelines.nn import MIMICPipeline as MIMICNNPipeline
from pipelines.regression import MIMICPipeline as MIMICRegPipeline
from model.sklearn.standard.linear_models import StandardLogReg
from model.tf2.logistic_regression import IncrementalLogReg

settings = json.load(Path(os.getenv("CONFIG"), "test.json").open())


def test_regression_pipeline_ihm():
    data_path = Path(os.getenv("WORKINGDIR"), "test", "temp")
    storage_path = Path(os.getenv("WORKINGDIR"), "test", "temp", "regression", "ihm")
    if storage_path.is_file():
        shutil.rmtree(str(storage_path))
    data_path = datasets.load_data(chunksize=5000000,
                                   source_path=TEST_DATA_GTRAW,
                                   storage_path=data_path,
                                   preprocess=True,
                                   engineer=True,
                                   task="IHM")
    pipeline_config = {
        'framework': 'sklearn',
        'output_type': 'sparse',
        'metrics': ['auc_roc', 'auc_pr'],
        'split_config': {
            'test_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "sklearn"),
        'task': 'in_hospital_mortality'
    }
    pipeline = MIMICRegPipeline(StandardLogReg('in_hospital_mortality'), **pipeline_config)
    pipeline.fit(data_path=data_path)

    pipeline_config = {
        'framework': 'tf2',
        'output_type': 'sparse',
        'metrics': ['auc_roc', 'auc_pr'],
        'patience': 4,
        'split_config': {
            'test_fraction_split': 0.2,
            'validation_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "tf2"),
        'task': 'in_hospital_mortality'
    }
    pipeline = MIMICRegPipeline(IncrementalLogReg('in_hospital_mortality'), **pipeline_config)
    pipeline.fit(data_path=data_path)


def test_regression_pipeline_phenotyping():
    data_path = Path(os.getenv("WORKINGDIR"), "test", "temp")
    storage_path = Path(os.getenv("WORKINGDIR"), "test", "temp", "regression", "PHENO")
    if storage_path.is_file():
        shutil.rmtree(str(storage_path))
    data_path = datasets.load_data(chunksize=5000000,
                                   source_path=TEST_DATA_GTRAW,
                                   storage_path=data_path,
                                   preprocess=True,
                                   engineer=True,
                                   task="PHENO")
    pipeline_config = {
        'framework': 'sklearn',
        'output_type': 'sparse',
        'metrics': ["auc_roc_micro", "auc_roc_macro"],
        'split_config': {
            'test_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "sklearn"),
        'task': "PHENO"
    }
    pipeline = MIMICRegPipeline(StandardLogReg("PHENO"), **pipeline_config)
    pipeline.fit(data_path=data_path)

    pipeline_config = {
        'framework': 'tf2',
        'output_type': 'sparse',
        'metrics': ["auc_roc_micro", "auc_roc_macro"],
        'patience': 4,
        'split_config': {
            'test_fraction_split': 0.2,
            'validation_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "tf2"),
        'task': "PHENO"
    }
    pipeline = MIMICRegPipeline(IncrementalLogReg("PHENO"), **pipeline_config)
    pipeline.fit(data_path=data_path)


def test_regression_pipeline_decomp():
    data_path = Path(os.getenv("WORKINGDIR"), "test", "temp")
    storage_path = Path(os.getenv("WORKINGDIR"), "test", "temp", "regression", "DECOMP")
    if storage_path.is_file():
        shutil.rmtree(str(storage_path))
    data_path = datasets.load_data(chunksize=5000000,
                                   source_path=TEST_DATA_GTRAW,
                                   storage_path=data_path,
                                   preprocess=True,
                                   engineer=True,
                                   task="DECOMP")
    """
    pipeline_config = {
        'framework': 'sklearn',
        'output_type': 'sparse',
        'metrics': ['auc_roc', 'auc_pr'],
        'split_config': {
            'test_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "sklearn"),
        'task': 'decompensation'
    }
    pipeline = MIMICRegPipeline(StandardLogReg('decompensation'), **pipeline_config)
    pipeline.fit(data_path=data_path)
    """
    pipeline_config = {
        'framework': 'tf2',
        'output_type': 'sparse',
        'metrics': ['auc_roc', 'auc_pr'],
        'patience': 4,
        'split_config': {
            'test_fraction_split': 0.2,
            'validation_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "tf2"),
        'task': 'decompensation'
    }
    pipeline = MIMICRegPipeline(IncrementalLogReg('decompensation'), **pipeline_config)
    pipeline.fit(data_path=data_path)


def test_regression_pipeline_los():
    data_path = Path(os.getenv("WORKINGDIR"), "test", "temp")
    storage_path = Path(os.getenv("WORKINGDIR"), "test", "temp", "regression", "LOS")
    if storage_path.is_file():
        shutil.rmtree(str(storage_path))
    data_path = datasets.load_data(chunksize=5000000,
                                   source_path=TEST_DATA_GTRAW,
                                   storage_path=data_path,
                                   preprocess=True,
                                   engineer=True,
                                   task="LOS")
    # One shot optimization not supported for los
    """
    pipeline_config = {
        'framework': 'sklearn',
        'output_type': 'sparse',
        'metrics': ['accuracy'],
        'split_config': {
            'test_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "sklearn"),
        'task': 'length_of_stay'
    }
    pipeline = MIMICRegPipeline(StandardLogReg('length_of_stay'), **pipeline_config)
    pipeline.fit(data_path=data_path)
    """

    pipeline_config = {
        'framework': 'tf2',
        'output_type': 'sparse',
        'metrics': ['accuracy'],
        'patience': 4,
        'split_config': {
            'test_fraction_split': 0.2,
            'validation_fraction_split': 0.2
        },
        'model_name': 'test',
        'root_path': Path(storage_path, "tf2"),
        'task': 'length_of_stay'
    }
    pipeline = MIMICRegPipeline(IncrementalLogReg('length_of_stay'), **pipeline_config)
    pipeline.fit(data_path=data_path)


if __name__ == "__main__":
    test_regression_pipeline_ihm()
    test_regression_pipeline_phenotyping()
    test_regression_pipeline_decomp()
    test_regression_pipeline_los()
'''
