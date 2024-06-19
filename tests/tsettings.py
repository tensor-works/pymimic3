import json
import os
from dotenv import load_dotenv
from pathlib import Path
from settings import *

load_dotenv(verbose=False)

__all__ = [
    'TEST_SETTINGS', 'SEMITEMP_DIR', 'TEMP_DIR', 'DEVTEMP_DIR', 'TEST_DATA_DIR', 'TEST_DATA_DEMO',
    'TEST_GT_DIR', 'TASK_NAMES', 'FTASK_NAMES', 'TASK_NAME_MAPPING'
]

TEST_SETTINGS = json.load(Path(os.getenv("TESTS"), "etc", "test.json").open())
SEMITEMP_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data", "semitemp")
TEMP_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data", "temp")
DEVTEMP_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data", "devtemp")
TEST_DATA_DIR = Path(os.getenv("WORKINGDIR"), "tests", "data")
TEST_DATA_DEMO = Path(TEST_DATA_DIR, "physionet.org", "files", "mimiciii-demo", "1.4")
TEST_GT_DIR = Path(TEST_DATA_DIR, "generated-benchmark")

FTASK_NAMES = [
    "in-hospital-mortality", "decompensation", "length-of-stay", "phenotyping", "multitask"
]
TASK_NAME_MAPPING = dict(zip(TASK_NAMES, FTASK_NAMES))
