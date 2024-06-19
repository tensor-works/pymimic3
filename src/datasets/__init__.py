"""
===================
Data Loading Module
===================

This module is responsible for extracting and processing data from the raw MIMIC-III dataset. The data 
extraction can be done using the `load_data` function, which supports various tasks for deep learning and 
machine learning, and can be customized using the different processors. By replicating the data pipeline
from the load_data() function and modifying the processors through inheritance granular steps, such as
feature engineering, can be customized. Data can be extracted in a single shot, similar to the original 
https://github.com/YerevaNN/mimic3-benchmarks github or in a multiprocessed, sped-up manner. The 
multiprocessed processing is tested against the originial extraction procedure.
The module also includes a comprehensive data split utility that allows for conventional splits as 
well as splits by demographic groups to induce concept drift.

CSV Files Used for Extraction
-----------------------------
The following CSV files from the MIMIC-III dataset are used in the data extraction and processing:

- ADMISSIONS.csv 
- CHARTEVENTS.csv
- DIAGNOSES_ICD.csv
- ICUSTAYS.csv 
- LABEVENTS.csv 
- OUTPUTS.csv
- PATIENTS.csv 
- D_ICD_DIAGNOSES.csv 
- D_ITEMS.csv
- DIAGNOSES_ICD.csv


Examples
--------
Example 1: Basic data loading
    >>> from data_extraction import load_data
    ...
    >>> dataset = load_data(source_path="/path/to/source", 
    ...                     storage_path="/path/to/storage")

Example 2: Data loading with preprocessing
    >>> from data_extraction import load_data
    ...
    >>> dataset = load_data(source_path="/path/to/source", 
    ...                     storage_path="/path/to/storage",
    ...                     preprocess=True, task="DECOMP")

Example 3: Iterative feature engineering with chunk size for river or sklearn algorithms
    >>> from data_extraction import load_data
    ...
    >>> dataset = load_data(source_path="/path/to/source", 
    ...                     storage_path="/path/to/storage",
    ...                     chunksize=1000, 
    ...                     extract=True, 
    ...                     preprocess=True, e
    ...                     ngineer=True,
    ...                     task="PHENO")
    
Example 3: Iterative feature engineering with chunk size for deep learning algorithms
    >>> from data_extraction import load_data
    ...
    >>> dataset = load_data(source_path="/path/to/source", 
    ...                     storage_path="/path/to/storage",
    ...                     chunksize=1000, 
    ...                     extract=True, 
    ...                     preprocess, 
    ...                     discretize=True,
    ...                     task="PHENO")

Module Functions
----------------
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
from datasets.mimic_utils import copy_subject_info

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
              deep_supervision: bool = False,
              extract: bool = True,
              preprocess: bool = False,
              engineer: bool = False,
              discretize: bool = False,
              task: str = None,
              verbose=True) -> Union[ProcessedSetReader, ExtractedSetReader, dict]:
    """
    Load and process the MIMIC-III dataset for machine learning and deep learning tasks.

    Parameters
    ----------
    source_path : str
        The location from which the unprocessed dataset is to be loaded. This should contain all the MIMIC-III
        CSV files from the physionet.org website and the resources folder which includes hcup_ccs_2015_definitions.yaml 
        and the itemid_to_variable_map.csv.
    storage_path : str, optional
        The location where the processed dataset is to be stored. If unspecifed, not iterative processing is
        possible and the dataset will be processed in RAM. Defaults to None.
    chunksize : int, optional
        The size of data chunks for iterative processing. Requires specified storage_path. Defaults to None.
    subject_ids : List[int], optional
        A list of subject IDs to be included in the dataset. If not specified all subject IDs are processed.
        Defaults to None.
    num_subjects : int, optional
        The number of subjects to be included in the dataset. If not specified all subject are processed.
        Defaults to None.
    time_step_size : float, optional
        The size of the time step for discretization. Defaults to 1.0(H).
    impute_strategy : str, optional
        The strategy for imputing missing values. Defaults to "previous". Can be either "'normal', 'previous', 'next', or 'zero'.
    mode : str, optional
        The mode of discretization. Can be either 'legacy' or 'experimental'. Defaults to "legacy".
    start_at_zero : bool, optional
        Whether to start time at zero or at the first timestamp. Defaults to True.
    extract : bool, optional
        Whether to perform data extraction. Defaults to True.
    preprocess : bool, optional
        Whether to perform data preprocessing. Defaults to False.
    engineer : bool, optional
        Whether to perform feature engineering. Defaults to False.
    discretize : bool, optional
        Whether to perform data discretization. Defaults to False.
    task : str, optional
        The specific task for which to process the data. Possible values are "DECOMP", "LOS", "IHM", "PHENO". Defaults to None.

    Returns
    -------
    Union[ProcessedSetReader, ExtractedSetReader, dict]
        The processed dataset or a reader if chunksize option is set, allowing to access the result dataset.

    Raises
    ------
    ValueError
        If invalid parameters are provided.

    Notes
    -----
    - Iterative generation is used if a chunksize is specified.
    - Compact generation is used otherwise.
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
                                                     task=task,
                                                     verbose=verbose)

        if preprocess or engineer or discretize:
            # Contains phenotypes and a list of codes referring to the phenotype
            with Path(source_path, "resources", "hcup_ccs_2015_definitions.yaml").open("r") as file:
                phenotypes_yaml = yaml.full_load(file)

            processed_storage_path = Path(storage_path, "processed", task)
            preprocessor = MIMICPreprocessor(task=task,
                                             storage_path=processed_storage_path,
                                             phenotypes_yaml=phenotypes_yaml,
                                             label_type="one-hot",
                                             verbose=verbose)
            copy_subject_info(reader.root_path, processed_storage_path)

            proc_reader = preprocessor.transform_reader(reader=reader,
                                                        subject_ids=subject_ids,
                                                        num_subjects=num_subjects)
            reader = proc_reader

        if engineer:
            engineered_storage_path = Path(storage_path, "engineered", task)
            engine = MIMICFeatureEngine(config_dict=Path(os.getenv("CONFIG"),
                                                         "engineering_config.json"),
                                        storage_path=engineered_storage_path,
                                        task=task,
                                        verbose=verbose)
            reader = engine.transform_reader(reader=proc_reader,
                                             subject_ids=subject_ids,
                                             num_subjects=num_subjects)
            copy_subject_info(proc_reader.root_path, engineered_storage_path)

        if discretize:
            discretized_storage_path = Path(storage_path, "discretized", task)
            discretizer = MIMICDiscretizer(task=task,
                                           storage_path=discretized_storage_path,
                                           time_step_size=time_step_size,
                                           impute_strategy=impute_strategy,
                                           start_at_zero=start_at_zero,
                                           mode=mode,
                                           deep_supervision=deep_supervision,
                                           verbose=verbose)
            reader = discretizer.transform_reader(reader=proc_reader,
                                                  subject_ids=subject_ids,
                                                  num_subjects=num_subjects)
            copy_subject_info(proc_reader.root_path, discretized_storage_path)

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
                                                task=task,
                                                verbose=verbose)
    if preprocess or engineer or discretize:
        processed_storage_path = Path(storage_path, "processed", task)
        # Contains phenotypes and a list of codes referring to the phenotype
        with Path(source_path, "resources", "hcup_ccs_2015_definitions.yaml").open("r") as file:
            phenotypes_yaml = yaml.full_load(file)
        preprocessor = MIMICPreprocessor(task=task,
                                         storage_path=processed_storage_path,
                                         phenotypes_yaml=phenotypes_yaml,
                                         label_type="one-hot",
                                         verbose=verbose)
        dataset = preprocessor.transform_dataset(dataset=dataset,
                                                 subject_ids=subject_ids,
                                                 num_subjects=num_subjects,
                                                 source_path=extracted_storage_path)
        copy_subject_info(extracted_storage_path, processed_storage_path)

    if engineer:
        engineered_storage_path = Path(storage_path, "engineered", task)
        engine = MIMICFeatureEngine(config_dict=Path(os.getenv("CONFIG"),
                                                     "engineering_config.json"),
                                    storage_path=engineered_storage_path,
                                    task=task,
                                    verbose=verbose)
        dataset = engine.transform_dataset(dataset,
                                           subject_ids=subject_ids,
                                           num_subjects=num_subjects)
        copy_subject_info(processed_storage_path, engineered_storage_path)

    if discretize:
        discretized_storage_path = Path(storage_path, "discretized", task)
        discretizer = MIMICDiscretizer(task=task,
                                       storage_path=discretized_storage_path,
                                       time_step_size=time_step_size,
                                       impute_strategy=impute_strategy,
                                       start_at_zero=start_at_zero,
                                       deep_supervision=deep_supervision,
                                       mode=mode,
                                       verbose=verbose)
        dataset = discretizer.transform_dataset(dataset=dataset,
                                                subject_ids=subject_ids,
                                                num_subjects=num_subjects)
        copy_subject_info(processed_storage_path, discretized_storage_path)

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
    if task == "MULTI" and engineer:
        raise ValueError("Task 'MULTI' is for DNN's only and feature extraction is not supported.")
    return None


if __name__ == "__main__":
    ...
