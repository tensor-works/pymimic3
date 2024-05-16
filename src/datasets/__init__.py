"""Dataset file

This file allows access to the dataset as specified.
All function in this file are used by the main interface function load_data.
Subfunctions used within private functions are located in the datasets.utils module.

Todo:
    - Use a settings.json
    - This is a construction site, see what you can bring in here
    - Provid link to kaggle in load_data doc string
    - Expand function to utils

YerevaNN/mimic3-benchmarks
"""
import yaml
from typing import Union
from pathlib import Path
from utils.IO import *
from settings import *
from . import extraction
from . import preprocessing
from . import feature_engineering
from . import discretizing
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
            reader = preprocessing.iterative_processing(reader=reader,
                                                        task=task,
                                                        subject_ids=subject_ids,
                                                        num_subjects=num_subjects,
                                                        storage_path=processed_storage_path,
                                                        phenotypes_yaml=phenotypes_yaml)

        if engineer:
            engineered_storage_path = Path(storage_path, "engineered", task)
            reader = feature_engineering.iterative_fengineering(
                subject_ids=subject_ids,
                num_subjects=num_subjects,
                reader=reader,
                task=task,
                storage_path=engineered_storage_path)

        if discretize:
            discretized_storage_path = Path(storage_path, "discretized", task)
            reader = discretizing.iterative_discretization(reader=reader,
                                                           task=task,
                                                           storage_path=discretized_storage_path,
                                                           time_step_size=time_step_size,
                                                           impute_strategy=impute_strategy,
                                                           start_at_zero=start_at_zero,
                                                           mode=mode)

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
        dataset = preprocessing.compact_processing(dataset=dataset,
                                                   task=task,
                                                   subject_ids=subject_ids,
                                                   num_subjects=num_subjects,
                                                   storage_path=processed_storage_path,
                                                   source_path=extracted_storage_path,
                                                   phenotypes_yaml=phenotypes_yaml)

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
    resource_folder = Path(os.getenv("WORKINGDIR"), "datalab", "mimic", "data_splits", "resources")
    handler = SplitHandler(Path(resource_folder, "subject_info_df.csv"),
                           Path(resource_folder, "progress.json"))
    assert len(handler.get_subjects("ETHNICITY", "WHITE")) == 30019
    assert len(handler.get_subjects("ETHNICITY", "BLACK")) == 3631
    assert len(handler.get_subjects("ETHNICITY", "UNKNOWN/NOT SPECIFIED")) == 3861
    assert len(handler.get_subjects("ETHNICITY", "HISPANIC")) == 1538
    assert len(handler.get_subjects("ETHNICITY", "ASIAN")) == 1623
    assert len(handler.get_subjects("ETHNICITY", "OTHER")) == 1902
    assert len(handler.get_subjects("ETHNICITY", "UNABLE TO OBTAIN")) == 703

    assert len(handler.get_subjects("ETHNICITY", ["WHITE", "BLACK"])) == 30019 + 3631
    assert len(handler.get_subjects("ETHNICITY", ["HISPANIC", "ASIAN"])) == 1538 + 1623

    dataset, ratios = handler.split("ETHNICITY", test="WHITE", train="BLACK")
    assert len(dataset["train"]) == 3631
    assert len(dataset["test"]) == 30019

    dataset, ratios = handler.split("ETHNICITY", test="WHITE")
    assert sum(list(ratios.values())) == 1
    assert len(dataset["train"]) == 3631 + 3861 + 1538 + 1623 + 1902 + 703
    assert len(dataset["test"]) == 30019

    dataset, ratios = handler.split("ETHNICITY", train="WHITE")
    assert sum(list(ratios.values())) == 1
    assert len(dataset["train"]) == 30019
    assert len(dataset["test"]) == 3631 + 3861 + 1538 + 1623 + 1902 + 703

    dataset, ratios = handler.split("LANGUAGE", train=["ENGL", "SPAN"])
    assert sum(list(ratios.values())) == 1
    assert len(dataset["train"]) == 19436 + 728
