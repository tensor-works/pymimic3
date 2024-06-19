import os
import sys

sys.path.append(os.getenv("WORKINGDIR"))
import pandas as pd
import numpy as np
from pathlib import Path
from tests.settings import TEST_DATA_DIR, TASK_NAMES
# This is copied into the mimic3benchmark directory once cloned
from mimic3benchmark.readers import InHospitalMortalityReader, DecompensationReader, LengthOfStayReader, PhenotypingReader
from mimic3models.in_hospital_mortality.logistic.main import read_and_extract_features as ihm_extractor
from mimic3models.phenotyping.logistic.main import read_and_extract_features as phenotyping_extractor
from mimic3models.length_of_stay.logistic.main import read_and_extract_features as los_extractor
from mimic3models.decompensation.logistic.main import read_and_extract_features as decompensation_extractor

# Set the paths to the data files
processed_paths = {
    "IHM": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "in-hospital-mortality"),
    "LOS": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "length-of-stay"),
    "PHENO": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "phenotyping"),
    "DECOMP": Path(TEST_DATA_DIR, "generated-benchmark", "processed", "decompensation")
}

engineered_paths = {
    "IHM": Path(TEST_DATA_DIR, "generated-benchmark", "engineered", "in-hospital-mortality"),
    "LOS": Path(TEST_DATA_DIR, "generated-benchmark", "engineered", "length-of-stay"),
    "PHENO": Path(TEST_DATA_DIR, "generated-benchmark", "engineered", "phenotyping"),
    "DECOMP": Path(TEST_DATA_DIR, "generated-benchmark", "engineered", "decompensation")
}

readers = {
    "IHM": InHospitalMortalityReader,
    "LOS": LengthOfStayReader,
    "PHENO": PhenotypingReader,
    "DECOMP": DecompensationReader
}

extractors = {
    "IHM": ihm_extractor,
    "LOS": los_extractor,
    "PHENO": phenotyping_extractor,
    "DECOMP": decompensation_extractor
}

# Create the readers for each task type
ihm_reader = InHospitalMortalityReader(dataset_dir=processed_paths["IHM"],
                                       listfile=Path(processed_paths["IHM"], "listfile.csv"))
decomp_reader = DecompensationReader(dataset_dir=processed_paths["DECOMP"],
                                     listfile=Path(processed_paths["DECOMP"], "listfile.csv"))
los_reader = LengthOfStayReader(dataset_dir=processed_paths["LOS"],
                                listfile=Path(processed_paths["LOS"], "listfile.csv"))
phenotyping_reader = PhenotypingReader(dataset_dir=processed_paths["PHENO"],
                                       listfile=Path(processed_paths["PHENO"], "listfile.csv"))

for task in TASK_NAMES:
    if engineered_paths[task].exists():
        continue
    if task in ["LOS", "DECOMP"]:
        print(f"Engineering data for task: {task}. This may take up to 30 min ...")
    else:
        print(f"Engineering data for task: {task}.")
    reader = readers[task](dataset_dir=processed_paths[task],
                           listfile=Path(processed_paths[task], "listfile.csv"))
    if task == "IHM":
        (X, y, train_names) = extractors[task](reader, period="all", features="all")
    elif task == "PHENO":
        (X, y, train_names, ts) = extractors[task](reader, period="all", features="all")
    else:
        n_samples = min(100000, reader.get_number_of_examples())
        (X, y, train_names, ts) = extractors[task](reader,
                                                   period="all",
                                                   features="all",
                                                   count=n_samples)

    print(f"Done engineering data for task: {task}.")
    # 10127_episode271544_timeseries.csv is included in the original DS despite the subject being a new born infant. Minimum age was set at 18
    X_df = pd.DataFrame(np.concatenate([np.array(train_names).reshape(-1, 1), X],
                                       axis=1)).set_index(0)
    if task == "IHM":
        y_df = pd.DataFrame(np.stack([np.array(train_names), y]).T).set_index(0)
    elif task == "PHENO":
        y_df = pd.DataFrame(
            np.concatenate([np.array(train_names).reshape(-1, 1),
                            np.array(y)], axis=1)).set_index(0)
    else:
        y_df = pd.DataFrame(
            np.concatenate([np.array(train_names).reshape(-1, 1),
                            np.array(y).reshape(-1, 1)],
                           axis=1)).set_index(0)
    engineered_paths[task].mkdir(parents=True, exist_ok=True)
    X_df.to_csv(str(Path(engineered_paths[task], "X.csv")))
    y_df.to_csv(str(Path(engineered_paths[task], "y.csv")))
