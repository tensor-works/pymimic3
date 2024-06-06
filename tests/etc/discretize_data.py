import os
import sys

sys.path.append(os.getenv("WORKINGDIR"))
import pandas as pd
from pathlib import Path
from tests.settings import *
from mimic3benchmark.readers import DecompensationReader

from mimic3models.preprocessing import Discretizer
from mimic3benchmark.readers import InHospitalMortalityReader, DecompensationReader, LengthOfStayReader, PhenotypingReader

# Set the paths to the data files
processed_paths = {
    "IHM": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "in-hospital-mortality"),
    "LOS": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "length-of-stay"),
    "PHENO": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "phenotyping"),
    "DECOMP": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "decompensation")
}

discretized_paths = {
    "IHM": Path(TEST_DATA_DIR, "generated-benchmark", "discretized", "in-hospital-mortality"),
    "LOS": Path(TEST_DATA_DIR, "generated-benchmark", "discretized", "length-of-stay"),
    "PHENO": Path(TEST_DATA_DIR, "generated-benchmark", "discretized", "phenotyping"),
    "DECOMP": Path(TEST_DATA_DIR, "generated-benchmark", "discretized", "decompensation")
}

readers = {
    "IHM": InHospitalMortalityReader,
    "LOS": LengthOfStayReader,
    "PHENO": PhenotypingReader,
    "DECOMP": DecompensationReader
}

impute_strategies = ['zero', 'normal_value', 'previous', 'next']
start_times = ['zero', 'relative']

# Discritize the data from processed directory using different discretizer settings
for task in TASK_NAMES:
    list_file_path = Path(processed_paths[task], "listfile.csv")
    list_file = pd.read_csv(
        list_file_path,
        na_values=[''],
        keep_default_na=False,
    )
    if task in ["IHM", "PHENO"]:
        example_indices = list_file.index
    else:
        example_indices = list_file.groupby("stay")["period_length"].idxmax().values
    discretized_paths[task].mkdir(parents=True, exist_ok=True)
    list_file.loc[example_indices].to_csv(Path(discretized_paths[task], "listfile.csv"),
                                          index=False)

    print(f"Discretizing {task} data")
    for impute_strategy in impute_strategies:
        for start_time in start_times:
            discretizer = Discretizer(timestep=1.0,
                                      store_masks=False,
                                      impute_strategy=impute_strategy,
                                      start_time=start_time)
            reader = readers[task](dataset_dir=processed_paths[task])
            for idx in example_indices:
                sample = reader.read_example(idx)
                discretized_data = discretizer.transform(sample['X'])
                discretizer_header = discretized_data[1].split(',')
                target_dir = Path(discretized_paths[task],
                                  f"imp{impute_strategy}_start{start_time}")
                target_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(discretized_data[0],
                             columns=discretizer_header).to_csv(Path(target_dir, sample['name']),
                                                                index=False)
