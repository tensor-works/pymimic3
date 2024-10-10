# This was used to reduce the MIMIC-III demo dataset to speed up CICD
import os
import pandas as pd
from pathlib import Path
from datasets.mimic_utils import convert_dtype_dict
from settings import *

# no mortality
no_mortality = [
    10017,
    20124,
    42458,
    10117,
    40601,
    41976,
    42321,
    10088,
    10119,
    41914,
]

# mortality
mortality = [
    10069,
    10019,
    10011,
    43909,
    10102,
    44154,
    10036,
    40503,
    10112,
    10111,
]

directory = Path(os.getenv("TESTS"), "data", "mimiciii-demo")

for entity in directory.iterdir():
    if entity.is_file() and entity.suffix == ".csv":
        if entity.stem in DATASET_SETTINGS:
            df = pd.read_csv(entity,
                             dtype=convert_dtype_dict(DATASET_SETTINGS[entity.stem]["dtype"]),
                             low_memory=False)
        else:
            df = pd.read_csv(entity, low_memory=False)
        if not 'SUBJECT_ID' in df.columns:
            continue
        df = df[df['SUBJECT_ID'].isin(no_mortality + mortality)]
        df.to_csv(entity, index=False)
