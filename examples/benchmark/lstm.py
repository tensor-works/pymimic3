import datasets
from pathlib import Path
from pipelines import Pipeline

for task_name in ["IHM", "LOS", "PHENO", "DECOMP"]:
    reader = datasets.load_data(chunksize=75836,
                                source_path=Path("data", "physionet.org", "files", "mimiciii-demo",
                                                 "1.4"),
                                storage_path=Path("temp"),
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task=task_name)
    datasets.train_test_split(reader=reader,
                              test_size=0.2,
                              val_size=0.2,
                              storage_path=Path("temp", "split"))
