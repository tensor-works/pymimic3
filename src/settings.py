import json
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.IO import *

load_dotenv(verbose=False)

__all__ = [
    'TASK_NAMES', 'DATASET_SETTINGS', 'DECOMP_SETTINGS', 'LOS_SETTINGS', 'PHENOT_SETTINGS',
    'IHM_SETTINGS'
]

TASK_NAMES = ["DECOMP", "LOS", "PHENO", "IHM"]

with Path(os.getenv("CONFIG"), "datasets.json").open() as file:
    DATASET_SETTINGS = json.load(file)
    DECOMP_SETTINGS = DATASET_SETTINGS["DECOMP"]
    LOS_SETTINGS = DATASET_SETTINGS["LOS"]
    PHENOT_SETTINGS = DATASET_SETTINGS["PHENO"]
    IHM_SETTINGS = DATASET_SETTINGS["IHM"]
