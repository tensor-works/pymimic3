import pandas as pd
from datasets.mimic_utils import upper_case_column_names
from tests.tsettings import *

for csv in TEST_DATA_DEMO.iterdir():
    if csv.is_dir() or csv.suffix != ".csv":
        continue

    df = pd.read_csv(csv,
                     dtype={
                         "ROW_ID": 'Int64',
                         "ICUSTAY_ID": 'Int64',
                         "HADM_ID": 'Int64',
                         "SUBJECT_ID": 'Int64',
                         "row_id": 'Int64',
                         "icustay_id": 'Int64',
                         "hadm_id": 'Int64',
                         "subject_id": 'Int64'
                     },
                     na_values=[''],
                     keep_default_na=False,
                     low_memory=False)
    df = upper_case_column_names(df)
    df.to_csv(csv, index=False)
