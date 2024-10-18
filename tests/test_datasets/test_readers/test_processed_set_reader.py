import datasets
import pytest
import pandas as pd
import numpy as np
from copy import deepcopy
from utils.IO import *
from tests.tsettings import *
from settings import *
from typing import Dict
from tests.pytest_utils.decorators import retry
from preprocessing.imputers import PartialImputer
from preprocessing.scalers import MinMaxScaler
from datasets.readers import ProcessedSetReader
from datasets.mimic_utils import upper_case_column_names

LABEL_COLS = {
    "IHM":
        "Y",
    "DECOMP":
        "Y",
    "LOS":
        "Y",
    "PHENO": [
        'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13',
        'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22', 'y23', 'y24'
    ],
    "MULTI": [
        'PHENO_LABELS', 'IHM_POS', 'IHM_MASK', 'IHM_LABEL', 'DECOMP_MASKS', 'DECOMP_LABELS',
        'LOS_MASKS', 'LOS_LABELS', 'LOS_VALUE'
    ]
}

MULTI_DTYPES = {
    "IHM_pos": "int",  # TODO this should be float
    "IHM_mask": "int",
    "IHM_label": "int",
    "DECOMP_masks": "object",
    "DECOMP_labels": "object",
    "LOS_masks": "object",
    "LOS_labels": "object",
    "LOS_value": "float",
    "PHENO_labels": "object"
}

DTYPES = {
    column_name.upper(): np.float64 if dtype == "float64" else object
    for column_name, dtype in DATASET_SETTINGS["timeseries"]["dtype"].items()
}

DISCRETIZED_COLUMNS = set([
    'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height',
    'Mean blood pressure', 'Oxygen saturation', 'pH', 'Respiratory rate', 'Systolic blood pressure',
    'Temperature', 'Weight', 'Capillary refill rate->0.0', 'Capillary refill rate->1.0',
    'Glascow coma scale eye opening->1 No Response', 'Glascow coma scale eye opening->2 To pain',
    'Glascow coma scale eye opening->3 To speech',
    'Glascow coma scale eye opening->4 Spontaneously', 'Glascow coma scale eye opening->To Pain',
    'Glascow coma scale eye opening->To Speech', 'Glascow coma scale eye opening->None',
    'Glascow coma scale eye opening->Spontaneously',
    'Glascow coma scale motor response->1 No Response',
    'Glascow coma scale motor response->2 Abnorm extensn',
    'Glascow coma scale motor response->3 Abnorm flexion',
    'Glascow coma scale motor response->4 Flex-withdraws',
    'Glascow coma scale motor response->5 Localizes Pain',
    'Glascow coma scale motor response->6 Obeys Commands',
    'Glascow coma scale motor response->Abnormal extension',
    'Glascow coma scale motor response->No response',
    'Glascow coma scale motor response->Localizes Pain',
    'Glascow coma scale motor response->Flex-withdraws',
    'Glascow coma scale motor response->Obeys Commands',
    'Glascow coma scale motor response->Abnormal Flexion', 'Glascow coma scale total->3',
    'Glascow coma scale total->4', 'Glascow coma scale total->5', 'Glascow coma scale total->6',
    'Glascow coma scale total->7', 'Glascow coma scale total->8', 'Glascow coma scale total->9',
    'Glascow coma scale total->10', 'Glascow coma scale total->11', 'Glascow coma scale total->12',
    'Glascow coma scale total->13', 'Glascow coma scale total->14', 'Glascow coma scale total->15',
    'Glascow coma scale verbal response->1 No Response',
    'Glascow coma scale verbal response->No Response',
    'Glascow coma scale verbal response->No Response-ETT',
    'Glascow coma scale verbal response->2 Incomp sounds',
    'Glascow coma scale verbal response->3 Inapprop words',
    'Glascow coma scale verbal response->4 Confused',
    'Glascow coma scale verbal response->Confused',
    'Glascow coma scale verbal response->5 Oriented',
    'Glascow coma scale verbal response->Oriented',
    'Glascow coma scale verbal response->Inappropriate Words',
    'Glascow coma scale verbal response->Incomprehensible sounds',
    'Glascow coma scale verbal response->1.0 ET/Trach'
])

# TODO! Warnings because we do not have timestamps anymore


@pytest.mark.parametrize("task_name", TASK_NAMES)
@pytest.mark.parametrize("reader_flavour", ["preprocessed", "discretized", "engineered"])
def test_read_sample(task_name: str, reader_flavour: str,
                     preprocessed_readers: Dict[str, ProcessedSetReader],
                     engineered_readers: Dict[str, ProcessedSetReader],
                     discretized_readers: Dict[str, ProcessedSetReader]):
    # Testing the most basic functionalities
    if reader_flavour == "preprocessed":
        reader = preprocessed_readers[task_name]
    elif reader_flavour == "discretized":
        reader = discretized_readers[task_name]
    elif reader_flavour == "engineered":
        if task_name == "MULTI":
            return
        reader = engineered_readers[task_name]  # Testing basic read function
    tests_io(f"Test case read sample for {reader_flavour} task {task_name}", level=0)
    # 10017: Single stay
    # 41976: Multiple stays

    # Testing without reading ids and timestamps
    check_sample(reader.read_sample(10017),
                 read_ids=False,
                 read_timestamps=False,
                 flavour=reader_flavour,
                 task_name=task_name)
    check_sample(reader.read_sample(41976),
                 read_ids=False,
                 read_timestamps=False,
                 flavour=reader_flavour,
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} without ids and timestamps passed")

    # Testing with reading ids and without timestamps
    check_sample(reader.read_sample(41976, read_ids=True),
                 read_ids=True,
                 read_timestamps=False,
                 flavour=reader_flavour,
                 task_name=task_name)
    check_sample(reader.read_sample(10017, read_ids=True),
                 read_ids=True,
                 read_timestamps=False,
                 flavour=reader_flavour,
                 task_name=task_name)
    tests_io(
        f"Suceeded testing read sample for task {task_name} with ids and without timestamps passed")

    # Testing convert to numpy on read
    check_sample(reader.read_sample(41976, data_type=np.ndarray),
                 read_ids=False,
                 read_timestamps=False,
                 flavour=reader_flavour,
                 data_type=np.ndarray,
                 task_name=task_name)

    check_sample(reader.read_sample(41976, data_type=pd.DataFrame),
                 read_ids=False,
                 read_timestamps=False,
                 flavour=reader_flavour,
                 data_type=pd.DataFrame,
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} with numpy conversion passed")


@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_read_sample_with_ds(task_name: str, discretized_readers: Dict[str, ProcessedSetReader]):
    # Testing deep supervision reads
    # 10017: Single stay
    # 41976: Multiple stays
    tests_io(
        f"Test case read discretized sample with deep supervision for discretized for task {task_name}",
        level=0)
    reader = discretized_readers[task_name]

    # Testing without ids with masks
    check_sample(reader.read_sample(10017, read_masks=True),
                 read_ids=False,
                 read_masks=True,
                 read_timestamps=False,
                 flavour="discretized",
                 task_name=task_name)
    check_sample(reader.read_sample(41976, read_masks=True),
                 read_ids=False,
                 read_masks=True,
                 read_timestamps=False,
                 flavour="discretized",
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} with deep supervision"
             " without ids or timestamps passed")

    # Testing with ids with masks
    check_sample(reader.read_sample(10017, read_masks=True, read_ids=True),
                 read_ids=True,
                 read_masks=True,
                 read_timestamps=False,
                 flavour="discretized",
                 task_name=task_name)
    check_sample(reader.read_sample(41976, read_masks=True, read_ids=True),
                 read_ids=True,
                 read_masks=True,
                 read_timestamps=False,
                 flavour="discretized",
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} with deep supervision"
             " with ids and without timestamps passed")


@pytest.mark.parametrize("task_name", set(TASK_NAMES) - set(["MULTI"]))
def test_read_sample_with_ts(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    tests_io(f"Test case read engineered samples with timestamp for engineerd for task {task_name}",
             level=0)
    reader = engineered_readers[task_name]
    # Testing with reading ids and timestamps
    check_sample(reader.read_sample(10017, read_ids=True, read_timestamps=True),
                 read_ids=True,
                 read_timestamps=True,
                 flavour="engineered",
                 task_name=task_name)
    check_sample(reader.read_sample(10017, read_ids=True, read_timestamps=True),
                 read_ids=True,
                 read_timestamps=True,
                 flavour="engineered",
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} with ids and timestamps passed")

    # Testing with timestamps but not ids
    check_sample(reader.read_sample(10017, read_ids=False, read_timestamps=True),
                 read_ids=False,
                 read_timestamps=True,
                 flavour="engineered",
                 task_name=task_name)
    check_sample(reader.read_sample(10017, read_ids=False, read_timestamps=True),
                 read_ids=False,
                 read_timestamps=True,
                 flavour="engineered",
                 task_name=task_name)
    tests_io(f"Suceeded testing read sample for task {task_name} "
             "with ids and without timestamps passed")


@pytest.mark.parametrize("task_name", TASK_NAMES)
@pytest.mark.parametrize("reader_flavour", ["preprocessed", "discretized", "engineered"])
def test_read_samples(task_name: str, reader_flavour: str,
                      preprocessed_readers: Dict[str, ProcessedSetReader],
                      engineered_readers: Dict[str, ProcessedSetReader],
                      discretized_readers: Dict[str, ProcessedSetReader]):
    # Testing the most basic functionalities
    if reader_flavour == "preprocessed":
        reader = preprocessed_readers[task_name]
    elif reader_flavour == "discretized":
        reader = discretized_readers[task_name]
    elif reader_flavour == "engineered":
        if task_name == "MULTI":
            return
        reader = engineered_readers[task_name]  # Testing basic read function
    tests_io(f"Test case read samples for {reader_flavour} for task {task_name}", level=0)
    # Testing without reading ids and timestamps
    check_samples(reader.read_samples([10017, 41976]),
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  task_name=task_name)
    check_samples(reader.read_samples([10017]),
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  task_name=task_name)
    check_samples(reader.read_samples(reader.subject_ids),
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  task_name=task_name)
    tests_io(f"Suceeded testing read samples for task {task_name}"
             " without ids and timestamps passed")

    # Testing with reading ids but without timestamps
    check_samples(reader.read_samples([10017, 41976], read_ids=True),
                  read_timestamps=False,
                  read_ids=True,
                  flavour=reader_flavour,
                  task_name=task_name)
    check_samples(reader.read_samples([10017], read_ids=True),
                  read_timestamps=False,
                  read_ids=True,
                  flavour=reader_flavour,
                  task_name=task_name)
    check_samples(reader.read_samples(reader.subject_ids, read_ids=True),
                  read_timestamps=False,
                  read_ids=True,
                  flavour=reader_flavour,
                  task_name=task_name)
    tests_io(f"Suceeded testing read samples for task"
             f" {task_name} with ids and without timestamps passed")

    # Testing convert to numpy on read
    check_samples(reader.read_samples([41976, 10017], data_type=np.ndarray),
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  data_type=np.ndarray,
                  task_name=task_name)

    check_samples(reader.read_samples([41976, 10017], data_type=pd.DataFrame),
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  data_type=pd.DataFrame,
                  task_name=task_name)
    tests_io(f"Suceeded testing read samples for task {task_name} with numpy conversion passed")


@pytest.mark.parametrize("task_name", set(TASK_NAMES) - set(["MULTI"]))
def test_read_samples_with_ts(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    tests_io(
        f"Test case read engineered samples with timestamp for engineered for task {task_name}",
        level=0)
    reader = engineered_readers[task_name]
    # Testing with reading ids and timestamps
    # Testing without reading IDs but with timestamps
    check_samples(reader.read_samples([10017, 41976], read_timestamps=True),
                  read_ids=False,
                  read_timestamps=True,
                  flavour="engineered",
                  task_name=task_name)

    check_samples(reader.read_samples([10017], read_timestamps=True),
                  read_timestamps=True,
                  read_ids=False,
                  flavour="engineered",
                  task_name=task_name)

    check_samples(reader.read_samples(reader.subject_ids, read_timestamps=True),
                  read_timestamps=True,
                  read_ids=False,
                  flavour="engineered",
                  task_name=task_name)
    tests_io(f"Succeeded in testing read samples without IDs and"
             f" with timestamps for task {task_name}")

    # Testing with reading IDs and timestamps
    check_samples(reader.read_samples([10017, 41976], read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  read_ids=True,
                  flavour="engineered",
                  task_name=task_name)

    check_samples(reader.read_samples([10017], read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  read_ids=True,
                  flavour="engineered",
                  task_name=task_name)

    check_samples(reader.read_samples(reader.subject_ids, read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  read_ids=True,
                  flavour="engineered",
                  task_name=task_name)
    tests_io(f"Succeeded in testing read samples with IDs and timestamps for task {task_name}")


@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_read_samples_with_ds(task_name: str, discretized_readers: Dict[str, ProcessedSetReader]):
    tests_io(
        f"Test case read deep supervision samples with timestamp for discretized for task {task_name}",
        level=0)
    # Testing without reading ids and timestamps
    reader = discretized_readers[task_name]

    # Testing without reading IDs and timestamps
    check_samples(reader.read_samples([10017, 41976], read_masks=True),
                  read_ids=False,
                  read_masks=True,
                  read_timestamps=False,
                  flavour="discretized",
                  task_name=task_name)
    check_samples(reader.read_samples([10017], read_masks=True),
                  read_ids=False,
                  read_masks=True,
                  read_timestamps=False,
                  flavour="discretized",
                  task_name=task_name)

    check_samples(reader.read_samples(reader.subject_ids, read_masks=True),
                  read_ids=False,
                  read_masks=True,
                  read_timestamps=False,
                  flavour="discretized",
                  task_name=task_name)
    tests_io(f"Succeeded testing read samples without IDs and timestamps")

    # Testing with reading IDs but without timestamps
    check_samples(reader.read_samples([10017, 41976], read_ids=True, read_masks=True),
                  read_ids=True,
                  read_timestamps=False,
                  read_masks=True,
                  flavour="discretized",
                  task_name=task_name)

    check_samples(reader.read_samples([10017], read_ids=True, read_masks=True),
                  read_ids=True,
                  read_timestamps=False,
                  read_masks=True,
                  flavour="discretized",
                  task_name=task_name)

    check_samples(reader.read_samples(reader.subject_ids, read_ids=True, read_masks=True),
                  read_ids=True,
                  read_timestamps=False,
                  read_masks=True,
                  flavour="discretized",
                  task_name=task_name)
    tests_io(f"Succeeded testing read samples with IDs but without timestamps")


@pytest.mark.parametrize("task_name", TASK_NAMES)
@pytest.mark.parametrize("reader_flavour", ["preprocessed", "discretized", "engineered"])
def test_random_samples(task_name: str, reader_flavour: str,
                        preprocessed_readers: Dict[str, ProcessedSetReader],
                        engineered_readers: Dict[str, ProcessedSetReader],
                        discretized_readers: Dict[str, ProcessedSetReader]):
    if reader_flavour == "preprocessed":
        reader = preprocessed_readers[task_name]
    elif reader_flavour == "discretized":
        reader = discretized_readers[task_name]
    elif reader_flavour == "engineered":
        if task_name == "MULTI":
            return
        reader = engineered_readers[task_name]

        # Testing without reading ids and timestamps

    tests_io(f"Test case random samples for {reader_flavour} task {task_name}", level=0)
    # Testing without reading ids and timestamps
    check_samples(reader.random_samples(10),
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  task_name=task_name)
    check_samples(reader.random_samples(),
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  task_name=task_name)
    tests_io(f"Succeeded testing random samples without IDs and timestamps for task {task_name}")

    # Testing with reading IDs but without timestamps
    check_samples(reader.random_samples(10, read_ids=True),
                  read_ids=True,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  task_name=task_name)
    check_samples(reader.random_samples(read_ids=True),
                  read_ids=True,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  task_name=task_name)
    tests_io(f"Succeeded testing random samples with IDs but without"
             f" timestamps for task {task_name}")
    # Test without replacement property
    samples = reader.random_samples(len(reader.subject_ids), read_ids=True)
    check_samples(samples=samples,
                  read_ids=True,
                  flavour=reader_flavour,
                  read_timestamps=False,
                  task_name=task_name)
    assert len(set(samples["X"].keys())) == len(samples["X"].keys())
    assert len(set(samples["X"].keys())) == len(reader.subject_ids)

    # Exceeding the set size results in warning and set sized sample collection
    samples = reader.random_samples(2 * len(reader.subject_ids), read_ids=True)
    check_samples(samples,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  read_ids=True,
                  task_name=task_name)
    assert len(set(samples["X"].keys())) == len(samples["X"].keys())
    assert len(set(samples["X"].keys())) == len(reader.subject_ids)
    tests_io(f"Suceeded testing random samples for task {task_name} without replacement passed")


@pytest.mark.parametrize("task_name", set(TASK_NAMES) - set(["MULTI"]))
def test_random_samples_with_ts(task_name: str, engineered_readers: Dict[str, ProcessedSetReader]):
    # Start of test case
    tests_io(f"Test case for random samples with timestamps for engineered for task {task_name}",
             level=0)

    reader = engineered_readers[task_name]

    # Testing with reading IDs and timestamps
    check_samples(reader.random_samples(10, read_ids=True, read_timestamps=True),
                  read_ids=True,
                  read_timestamps=True,
                  flavour="engineered",
                  task_name=task_name)

    check_samples(reader.random_samples(read_ids=True, read_timestamps=True),
                  read_timestamps=True,
                  read_ids=True,
                  flavour="engineered",
                  task_name=task_name)

    tests_io(f"Succeeded testing random samples for task {task_name} with IDs and timestamps")

    # Testing with reading timestamps only
    check_samples(reader.random_samples(10, read_timestamps=True),
                  read_ids=False,
                  read_timestamps=True,
                  flavour="engineered",
                  task_name=task_name)

    check_samples(reader.random_samples(read_timestamps=True),
                  read_timestamps=True,
                  read_ids=False,
                  flavour="engineered",
                  task_name=task_name)

    tests_io(f"Succeeded testing random samples for task {task_name} with timestamps only")

    tests_io(f"Finished test case for random samples with timestamps for task {task_name}")


@pytest.mark.parametrize("task_name", ["DECOMP", "LOS"])
def test_random_samples_with_ds(task_name: str, discretized_readers: Dict[str, ProcessedSetReader]):
    tests_io(
        f"Test case for random samples with deep supervision for discretized for task {task_name}",
        level=0)

    reader = discretized_readers[task_name]

    # Testing random samples with masks (no IDs)
    samples = reader.random_samples(10, read_masks=True)
    check_samples(samples=samples,
                  read_ids=False,
                  read_masks=True,
                  flavour="discretized",
                  read_timestamps=False,
                  task_name=task_name)
    tests_io(f"Succeeded in testing random samples with masks only for task {task_name}")

    tests_io(f"Succeeded in testing all random samples with masks only for task {task_name}")
    samples = reader.random_samples(len(reader.subject_ids), read_masks=True)
    check_samples(samples=samples,
                  read_ids=False,
                  read_masks=True,
                  flavour="discretized",
                  read_timestamps=False,
                  task_name=task_name)

    # Testing random samples with masks and IDs
    tests_io(f"Succeeded in testing all random samples with masks and IDs for task {task_name}")
    samples = reader.random_samples(len(reader.subject_ids), read_ids=True, read_masks=True)
    check_samples(samples=samples,
                  read_ids=True,
                  read_masks=True,
                  flavour="discretized",
                  read_timestamps=False,
                  task_name=task_name)

    tests_io(f"Succeeded in testing random samples with masks and IDs for task {task_name}")
    samples = reader.random_samples(10, read_masks=True, read_ids=True)
    check_samples(samples=samples,
                  read_ids=True,
                  read_masks=True,
                  flavour="discretized",
                  read_timestamps=False,
                  task_name=task_name)

    tests_io(f"Succeeded testing random samples with deep "
             f"supervision for task {task_name} with masks")


@pytest.mark.parametrize("task_name", TASK_NAMES)
@pytest.mark.parametrize("reader_flavour", ["discretized", "engineered"])
@retry(2)
def test_to_numpy(task_name: str, reader_flavour: str,
                  discretized_readers: Dict[str, ProcessedSetReader],
                  engineered_readers: Dict[str, ProcessedSetReader]):

    # TODO! Implement multi readers
    if task_name == "MULTI":
        return

    # Imputation is done inside the discretizer
    if reader_flavour == "discretized":
        reader = discretized_readers[task_name]
        imputer = None
    # No MULTI for engineered since its only used by DNNs
    elif reader_flavour == "engineered":
        if task_name == "MULTI":
            return
        reader = engineered_readers[task_name]
        imputer = PartialImputer().fit_reader(reader)

    # When running on reduced dataset, there are only 18 subjects for IHM and PHENO
    n_samples = 200
    if task_name in ["IHM", "PHENO"]:
        n_samples = 10

    # Prepare
    tests_io(f"Test case for to_numpy for task {task_name}", level=0)
    scaler = MinMaxScaler(imputer=imputer).fit_reader(reader)

    # => Read the samples
    dataset = reader.to_numpy(n_samples, scaler=scaler, imputer=imputer, bining="custom")

    # => Assert n samples (subjects)
    for prefix in dataset:
        assert np.isclose(dataset[prefix].shape[0] == n_samples, atol=1)

    tests_io(f"Succeeded in testing retriving limited amount of samples"
             f" to_numpy for {reader_flavour} for task {task_name}")

    # => Check sample content
    for prefix in dataset:
        dataset[prefix] = [dataset[prefix][index] for index in range(dataset[prefix].shape[0])]

    check_samples(dataset,
                  read_ids=False,
                  read_timestamps=False,
                  flavour=reader_flavour,
                  data_type=np.ndarray,
                  task_name=task_name)

    tests_io(f"Succeeded in testing retrived sample sanity for"
             f" to_numpy for {reader_flavour} for task {task_name}")


def check_samples(samples: dict,
                  read_timestamps: bool,
                  flavour: str,
                  read_ids: bool,
                  task_name: str,
                  read_masks: bool = False,
                  data_type=None):
    assert isinstance(samples, dict)
    base_keys = set(["X"])
    if read_timestamps:
        base_keys.add("t")
    if read_masks:
        base_keys.add("M")
        y_key = "yds"
    else:
        y_key = "y"
    base_keys.add(y_key)
    assert set(samples.keys()) == base_keys
    for index_or_id in samples["X"] if read_ids else range(len(samples["X"])):
        assert isinstance(index_or_id, int)
        sample = {
            key: samples[key][index_or_id] if read_ids else [samples[key][index_or_id]]
            for key in base_keys
        }
        check_sample(sample,
                     flavour=flavour,
                     read_ids=read_ids,
                     task_name=task_name,
                     read_timestamps=read_timestamps,
                     read_masks=read_masks,
                     data_type=data_type)


def check_sample(sample: dict,
                 read_ids: bool,
                 read_timestamps: bool,
                 flavour: str,
                 task_name: str,
                 read_masks: bool = False,
                 data_type=None):
    base_keys = set(["X"])
    if read_timestamps:
        base_keys.add("t")
    if read_masks:
        base_keys.add("M")
        base_keys.add("yds")
        y_key = "yds"
    else:
        base_keys.add("y")
        y_key = "y"
    assert set(sample.keys()) == base_keys
    X, Y = sample["X"], sample[y_key]

    if read_ids:
        assert isinstance(X, dict)
        assert isinstance(Y, dict)
        assert len(X) == len(Y)
        assert all([
            isinstance(stay_id, int) and
            (isinstance(timeseries, np.ndarray) if data_type == np.ndarray else isinstance(
                timeseries, pd.DataFrame)) for stay_id, timeseries in X.items()
        ])
    else:
        assert isinstance(X, list)
        assert isinstance(Y, list)
        assert len(X) == len(Y)
        assert all([(isinstance(timeseries, np.ndarray) if data_type == np.ndarray else isinstance(
            timeseries, pd.DataFrame)) for timeseries in X])
    for index_or_id in X.keys() if read_ids else range(len(X)):
        X_sample = X[index_or_id]
        Y_sample = Y[index_or_id]
        M_sample = sample["M"][index_or_id] if read_masks else False
        t_sample = sample["t"][index_or_id] if read_timestamps else False
        if data_type == pd.DataFrame or data_type is None:
            check_pd(X_sample, Y_sample, t_sample, M_sample, flavour, task_name)
        else:
            check_numpy(X_sample, Y_sample, t_sample, M_sample, flavour, task_name)


def check_pd(X_sample: pd.DataFrame, Y_sample: pd.DataFrame, t_sample: pd.DataFrame,
             M_sample: pd.DataFrame, flavour: str, task_name: str):
    if flavour == "discretized":
        base_columns = DISCRETIZED_COLUMNS
        X_sample = upper_case_column_names(deepcopy(X_sample))
    elif flavour == "engineered":
        base_columns = list(range(714))
    elif flavour == "preprocessed":
        base_columns = DATASET_SETTINGS["timeseries"]["dtype"].keys()
        X_sample = upper_case_column_names(deepcopy(X_sample))
    assert set([column.upper() if isinstance(column, str) else column for column in base_columns
               ]) == set(X_sample)

    if flavour in ["engineered", "discretized"]:
        assert all(X_sample.dtypes.apply(pd.api.types.is_float_dtype))
        if task_name == "MULTI":
            ...
            # TODO!
            # assert all(Y_sample.dtypes == pd.Series(MULTI_DTYPES))
        else:
            assert all(Y_sample.dtypes.apply(pd.api.types.is_float_dtype)) | \
                   all(Y_sample.dtypes.apply(pd.api.types.is_integer_dtype))
        if isinstance(M_sample, pd.DataFrame):
            assert all(M_sample.dtypes.apply(pd.api.types.is_bool_dtype))

            assert len(Y_sample) == len(M_sample)
        if isinstance(t_sample, pd.DataFrame):
            assert all(t_sample.dtypes.apply(pd.api.types.is_float_dtype))

            assert len(Y_sample) == len(t_sample)
    else:
        assert set([column.upper() for column in LABEL_COLS[task_name]
                   ]) == set(Y_sample.columns.str.upper())

        assert all([X_sample[column].dtype == DTYPES[column] for column in X_sample.columns])


def check_numpy(X_sample: pd.DataFrame, Y_sample: pd.DataFrame, t_sample: pd.DataFrame,
                M_sample: pd.DataFrame, flavour: str, task_name: str):
    assert isinstance(X_sample, np.ndarray)
    assert isinstance(Y_sample, np.ndarray)
    if flavour == "engineered":
        # N engineered features
        assert X_sample.shape[1] == 714
        if isinstance(t_sample, pd.DataFrame):
            assert len(t_sample) == len(Y_sample)
    elif flavour == "discretized":
        # N features after categorization
        assert X_sample.shape[1] == 59
        if isinstance(M_sample, pd.DataFrame):
            assert len(Y_sample) == len(M_sample)
    else:
        # N features raw
        assert X_sample.shape[1] == len(DATASET_SETTINGS["timeseries"]["dtype"])
    assert Y_sample.shape[1] == len(LABEL_COLS[task_name])


if __name__ == "__main__":
    import shutil
    # if SEMITEMP_DIR.is_dir():
    #     shutil.rmtree(str(SEMITEMP_DIR))
    proc_reader_dict = dict()
    eng_reader_dict = dict()
    disc_reader_dict = dict()
    for task_name in TASK_NAMES:
        proc_reader = datasets.load_data(chunksize=75835,
                                         source_path=TEST_DATA_DEMO,
                                         storage_path=SEMITEMP_DIR,
                                         preprocess=True,
                                         task=task_name)

        disc_reader = datasets.load_data(chunksize=75835,
                                         source_path=TEST_DATA_DEMO,
                                         storage_path=SEMITEMP_DIR,
                                         discretize=True,
                                         task=task_name)

        proc_reader_dict[task_name] = proc_reader
        disc_reader_dict[task_name] = disc_reader
        if task_name != "MULTI":
            eng_reader = datasets.load_data(chunksize=75835,
                                            source_path=TEST_DATA_DEMO,
                                            storage_path=SEMITEMP_DIR,
                                            engineer=True,
                                            task=task_name)
            eng_reader_dict[task_name] = eng_reader
            test_random_samples_with_ts(task_name, eng_reader_dict)
            test_read_sample_with_ts(task_name, eng_reader_dict)
            test_read_samples_with_ts(task_name, eng_reader_dict)
        else:
            eng_reader_dict[task_name] = None
        if task_name in ["DECOMP", "LOS"]:
            disc_reader = datasets.load_data(chunksize=75835,
                                             source_path=TEST_DATA_DEMO,
                                             storage_path=SEMITEMP_DIR,
                                             discretize=True,
                                             deep_supervision=True,
                                             task=task_name)
            test_random_samples_with_ds(task_name, disc_reader_dict)
            test_read_sample_with_ds(task_name, disc_reader_dict)
            test_read_samples_with_ds(task_name, disc_reader_dict)
        for flavour in ["preprocessed", "engineered", "discretized"]:
            test_random_samples(task_name, flavour, proc_reader_dict, eng_reader_dict,
                                disc_reader_dict)
            test_read_sample(task_name, flavour, proc_reader_dict, eng_reader_dict,
                             disc_reader_dict)
            test_read_samples(task_name, flavour, proc_reader_dict, eng_reader_dict,
                              disc_reader_dict)
            ...
        for flavour in ["engineered", "discretized"]:
            test_to_numpy(task_name, flavour, disc_reader_dict, eng_reader_dict)

    print("All tests passed!")
    # if TEMP_DIR.is_dir():
    #     shutil.rmtree(str(TEMP_DIR))
