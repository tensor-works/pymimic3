import json
import os
from pathlib import Path

__all__ = [
    'TEST_SETTINGS', 'SEMITEMP_DIR', 'TEMP_DIR', 'DEVTEMP_DIR', 'TEST_DATA_DIR', 'TEST_DATA_DEMO',
    'TEST_GT_DIR', 'TASK_NAMES', 'FTASK_NAMES', 'TASK_NAME_MAPPING', 'DATASET_SETTINGS',
    'DECOMP_SETTINGS', 'LOS_SETTINGS', 'PHENOT_SETTINGS', 'IHM_SETTINGS'
]

TEST_SETTINGS = json.load(Path(os.getenv("TESTS"), "etc", "test.json").open())
SEMITEMP_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data", "semitemp")
TEMP_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data", "temp")
DEVTEMP_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data", "devtemp")
TEST_DATA_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data")
TEST_DATA_DEMO = Path(TEST_DATA_DIR, "physionet.org", "files", "mimiciii-demo", "1.4")
TEST_GT_DIR = Path(TEST_DATA_DIR, "generated-benchmark")

TASK_NAMES = ["IHM", "DECOMP", "LOS", "PHENO"]
FTASK_NAMES = ["in-hospital-mortality", "decompensation", "length-of-stay", "phenotyping"]
TASK_NAME_MAPPING = dict(zip(TASK_NAMES, FTASK_NAMES))

with Path(os.getenv("CONFIG"), "datasets.json").open() as file:
    DATASET_SETTINGS = json.load(file)
    DECOMP_SETTINGS = DATASET_SETTINGS["DECOMP"]
    LOS_SETTINGS = DATASET_SETTINGS["LOS"]
    PHENOT_SETTINGS = DATASET_SETTINGS["PHENO"]
    IHM_SETTINGS = DATASET_SETTINGS["IHM"]
