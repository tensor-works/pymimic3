import json
import os
import bisect
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from utils.IO import *

load_dotenv(verbose=False)

__all__ = [
    'TASK_NAMES', 'DATASET_SETTINGS', 'DECOMP_SETTINGS', 'LOS_SETTINGS', 'PHENOT_SETTINGS',
    'IHM_SETTINGS', 'TEXT_METRICS', 'LOS_BINS', 'LOS_MEANS'
]

TASK_NAMES = ["DECOMP", "LOS", "PHENO", "IHM"]
TEXT_METRICS = ["classification_report", "confusion_matrix"]
LOS_BINS = [(-np.inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14),
            (14, np.inf)]
LOS_MEANS = [
    11.450379, 35.070846, 59.206531, 83.382723, 107.487817, 131.579534, 155.643957, 179.660558,
    254.306624, 585.325890
]

with Path(os.getenv("CONFIG"), "datasets.json").open() as file:
    DATASET_SETTINGS = json.load(file)
    DECOMP_SETTINGS = DATASET_SETTINGS["DECOMP"]
    LOS_SETTINGS = DATASET_SETTINGS["LOS"]
    PHENOT_SETTINGS = DATASET_SETTINGS["PHENO"]
    IHM_SETTINGS = DATASET_SETTINGS["IHM"]
