import json
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.IO import *

load_dotenv(verbose=False)

__all__ = [
    'TASK_NAMES', 'DATASET_SETTINGS', 'DECOMP_SETTINGS', 'LOS_SETTINGS', 'PHENOT_SETTINGS',
    'IHM_SETTINGS', 'TEXT_METRICS', 'MULTI_SETTINGS'
]

TASK_NAMES = ["IHM", "DECOMP", "LOS", "PHENO", "MULTI"]
TEXT_METRICS = ["classification_report", "confusion_matrix"]

with Path(os.getenv("CONFIG"), "datasets.json").open() as file:
    DATASET_SETTINGS = json.load(file)
    DECOMP_SETTINGS = DATASET_SETTINGS["DECOMP"]
    LOS_SETTINGS = DATASET_SETTINGS["LOS"]
    PHENOT_SETTINGS = DATASET_SETTINGS["PHENO"]
    IHM_SETTINGS = DATASET_SETTINGS["IHM"]
    MULTI_SETTINGS = DATASET_SETTINGS["MULTI"]
