import os
from pathlib import Path
import pandas as pd
import sys

sys.path.append(os.getenv("WORKINGDIR"))
from tests.settings import *

result_paths = [
    Path(TEST_DATA_DIR, "generated-benchmark", "extracted"),
    Path(TEST_DATA_DIR, "generated-benchmark", "processed", "in-hospital-mortality"),
    Path(TEST_DATA_DIR, "generated-benchmark", "processed", "length-of-stay"),
    Path(TEST_DATA_DIR, "generated-benchmark", "processed", "multitask"),
    Path(TEST_DATA_DIR, "generated-benchmark", "processed", "phenotyping"),
    Path(TEST_DATA_DIR, "generated-benchmark", "processed", "decompensation"),
]

for path in result_paths:
    for entity in path.iterdir():
        if entity.is_dir():
            for subject_entity in entity.iterdir():
                if subject_entity.is_dir():
                    for csv in subject_entity.iterdir():
                        target_path = Path(csv.parents[2], csv.parent.name, csv.name)
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        csv.rename(target_path)
                    subject_entity.rmdir()
                else:
                    target_path = Path(subject_entity.parents[1], subject_entity.name)
                    if subject_entity.name == "listfile.csv" and target_path.exists():
                        listfile_df = pd.read_csv(subject_entity)
                        listfile_df.to_csv(target_path, mode='a', index=False, header=False)
                        subject_entity.unlink()
                    else:
                        subject_entity.rename(target_path)
            entity.rmdir()
