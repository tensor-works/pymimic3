import pdb
import re
import os
import pandas as pd
from pathlib import Path
import sys
sys.path.append(os.getenv("WORKINGDIR"))
from tests.settings import *


for subject_dir in Path(TEST_DATA_DIR, "generated-benchmark", "extracted").iterdir():
    if not subject_dir.is_dir():
        continue
    for csv in subject_dir.iterdir():
        if "episode" in csv.name and not "timeseries" in csv.name:
            df = pd.read_csv(csv)
            icustay_id = df["Icustay"].iloc[0]
            file_index = re.findall(r'\d+', csv.name).pop()
            csv.rename(Path(csv.parent, f"episode{icustay_id}.csv"))
            Path(csv.parent, f"episode{file_index}_timeseries.csv").rename(Path(csv.parent, f"episode{icustay_id}_timeseries.csv"))
            # I want to rename the files episodeX.csv and episodeX_timeseries.csv by replacing X, which in this case is a random number by the icustay
