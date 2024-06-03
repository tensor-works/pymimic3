"""Dataset file

This file allows access to the dataset as specified.
All function in this file are used by the main interface function load_data.
Subfunctions used within private functions are located in the datasets.utils module.

TODOS
- Use a settings.json
- This is a construction site, see what you can bring in here
- Provid link to kaggle in load_data doc string
- Expand function to utils

YerevaNN/mimic3-benchmarks
"""
import yaml
import os
from typing import Union
from pathlib import Path
from utils.IO import *
from settings import *
from datasets.processors.preprocessors import MIMICPreprocessor
from datasets.processors.feature_engines import MIMICFeatureEngine
from datasets.processors.discretizers import MIMICDiscretizer
from . import extraction
from .readers import ProcessedSetReader, ExtractedSetReader
from .split import train_test_split

# global settings

__all__ = ["load_data", "train_test_split"]


def load_data(source_path: str,
              storage_path: str = None,
              chunksize: int = None,
              subject_ids: list = None,
              num_subjects: int = None,
              time_step_size: float = 1.0,
              impute_strategy: str = "previous",
              mode: str = "legacy",
              start_at_zero=True,
              extract: bool = True,
              preprocess: bool = False,
              engineer: bool = False,
              discretize: bool = False,
              task: str = None) -> Union[ProcessedSetReader, ExtractedSetReader, dict]:
    """_summary_

    Args:
        stoarge_path (str, optional): Location where the processed dataset is to be stored. Defaults to None.
        source_path (str, optional): Location form which the unprocessed dataset is to be loaded. Defaults to None.
        ehr (str, optional): _description_. Defaults to None.
        from_storage (bool, optional): _description_. Defaults to True.
        chunksize (int, optional): _description_. Defaults to None.
        num_subjects (int, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    storage_path = Path(storage_path)
    source_path = Path(source_path)

    subject_ids = _check_inputs(storage_path=storage_path,
                                source_path=source_path,
                                chunksize=chunksize,
                                subject_ids=subject_ids,
                                num_subjects=num_subjects,
                                extract=extract,
                                preprocess=preprocess,
                                engineer=engineer,
                                discretize=discretize,
                                task=task)

    # Iterative generation if a chunk size is specified
    if storage_path is not None and chunksize is not None:
        if extract or preprocess or engineer or discretize:
            extracted_storage_path = Path(storage_path, "extracted")
            # Account for missing subjects
            reader = extraction.iterative_extraction(storage_path=extracted_storage_path,
                                                     source_path=source_path,
                                                     chunksize=chunksize,
                                                     subject_ids=subject_ids,
                                                     num_subjects=num_subjects,
                                                     task=task)

        if preprocess or engineer or discretize:
            # Contains phenotypes and a list of codes referring to the phenotype
            with Path(source_path, "resources", "hcup_ccs_2015_definitions.yaml").open("r") as file:
                phenotypes_yaml = yaml.full_load(file)

            processed_storage_path = Path(storage_path, "processed", task)
            preprocessor = MIMICPreprocessor(task=task,
                                            storage_path=processed_storage_path,
                                            phenotypes_yaml=phenotypes_yaml,
                                            label_type="one-hot",
                                            verbose=True)
            
            reader = preprocessor.transform_reader(reader=reader, 
                                                   subject_ids=subject_ids, 
                                                   num_subjects=num_subjects)

        if engineer:
            engineered_storage_path = Path(storage_path, "engineered", task)
            engine = MIMICFeatureEngine(config_dict=Path(os.getenv("CONFIG"), "engineering_config.json"),
                                        storage_path=storage_path,
                                        task=task,
                                        verbose=True)
            reader = engine.transform_reader(reader=reader,
                                            subject_ids=subject_ids,
                                            num_subjects=num_subjects)
        if discretize:
            discretized_storage_path = Path(storage_path, "discretized", task)
            discretizer = MIMICDiscretizer(reader=reader,
                                        task=task,
                                        storage_path=discretized_storage_path,
                                        time_step_size=time_step_size,
                                        impute_strategy=impute_strategy,
                                        start_at_zero=start_at_zero,
                                        mode=mode,
                                        verbose=False)
            reader = discretizer.transform_reader(reader=reader,
                                                subject_ids=subject_ids,
                                                num_subjects=num_subjects)
            '''
            reader = discretizing.iterative_discretization(reader=reader,
                                                           task=task,
                                                           storage_path=discretized_storage_path,
                                                           time_step_size=time_step_size,
                                                           impute_strategy=impute_strategy,
                                                           start_at_zero=start_at_zero,
                                                           mode=mode)
            '''

        return reader

    elif chunksize is not None:
        raise ValueError("To run iterative iteration, specify storage path!")

    # Compact generation otherwise
    if extract or preprocess or engineer or discretize:
        extracted_storage_path = Path(storage_path, "extracted")
        dataset = extraction.compact_extraction(storage_path=extracted_storage_path,
                                                source_path=source_path,
                                                num_subjects=num_subjects,
                                                subject_ids=subject_ids,
                                                task=task)
    if preprocess or engineer or discretize:
        processed_storage_path = Path(storage_path, "processed", task)
        # Contains phenotypes and a list of codes referring to the phenotype
        with Path(source_path, "resources", "hcup_ccs_2015_definitions.yaml").open("r") as file:
            phenotypes_yaml = yaml.full_load(file)
        preprocessor = MIMICPreprocessor(task=task,
                                            storage_path=processed_storage_path,
                                            phenotypes_yaml=phenotypes_yaml,
                                            label_type="one-hot",
                                            verbose=True)
        dataset = preprocessor.transform_dataset(dataset=dataset,
                                       subject_ids=subject_ids,
                                       num_subjects=num_subjects,
                                       source_path=extracted_storage_path)
        '''
        dataset = preprocessing.compact_processing(dataset=dataset,
                                                   task=task,
                                                   subject_ids=subject_ids,
                                                   num_subjects=num_subjects,
                                                   storage_path=processed_storage_path,
                                                   source_path=extracted_storage_path,
                                                   phenotypes_yaml=phenotypes_yaml)
        '''

    if engineer:
        engineered_storage_path = Path(storage_path, "engineered", task)
        dataset = feature_engineering.compact_fengineering(dataset["X"],
                                                           dataset["y"],
                                                           task=task,
                                                           storage_path=engineered_storage_path,
                                                           source_path=processed_storage_path,
                                                           subject_ids=subject_ids,
                                                           num_subjects=num_subjects)

    if discretize:
        discretized_storage_path = Path(storage_path, "discretized", task)
        dataset = discretizing.compact_discretization(dataset["X"],
                                                      dataset["y"],
                                                      task=task,
                                                      storage_path=discretized_storage_path,
                                                      source_path=processed_storage_path,
                                                      time_step_size=time_step_size,
                                                      impute_strategy=impute_strategy,
                                                      start_at_zero=start_at_zero,
                                                      mode=mode)

    # TODO: make dependent from return reader (can also return reader)
    # TODO: write some tests for comparct generation
    return dataset


def _check_inputs(storage_path: str, source_path: str, chunksize: int, subject_ids: list,
                  num_subjects: int, extract: bool, preprocess: bool, engineer: bool,
                  discretize: bool, task: str):
    if chunksize and not storage_path:
        raise ValueError(f"Specify storage path if using iterative processing!"
                         f"Storage path is '{storage_path}' and chunksize is '{chunksize}'")
    if (preprocess or engineer) and not task:
        raise ValueError(
            "Specify the 'task' parameter for which to preprocess or engineer the data!"
            " Possible values for task are: DECOMP, LOS, IHM, PHENO")
    if task and not (engineer or preprocess or discretize):
        warn_io(f"Specified  task '{task}' for data extraction only, despite "
                "data extraction being task agnostic. Parameter is ignored.")
    if subject_ids and num_subjects:
        raise ValueError("Specify either subject_ids or num_subjects, not both!")
    if not any([extract, preprocess, engineer]):
        raise ValueError("One of extract, preprocess or engineer must be set to load the dataset.")
    if subject_ids is not None:
        return [int(subject_id) for subject_id in subject_ids]
    return None


if __name__ == "__main__":
    ...
