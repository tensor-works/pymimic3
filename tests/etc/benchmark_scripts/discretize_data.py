import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore, Style
from yaspin import yaspin
from tests.pytest_utils.reporting import nullyaspin

load_dotenv()

TASK_NAMES = ["IHM", "LOS", "PHENO", "DECOMP", "MULTI"]

if __name__ == "__main__":
    processedDir = Path(sys.argv[1])
    discretizedDir = Path(sys.argv[2])
    repositoryDir = Path(sys.argv[3])
    # processedDir = Path(os.getenv("TESTS"), "data", "control-dataset", "processed")
    # discretizedDir = Path(os.getenv("TESTS"), "data", "control-dataset", "engineered")
    # repositoryDir = Path(os.getenv("TESTS"), "data", "mimic3benchmarks")

    sys.path.append(str(repositoryDir))

    from mimic3benchmark.readers import DecompensationReader
    from mimic3models.preprocessing import Discretizer
    from mimic3benchmark.readers import InHospitalMortalityReader, DecompensationReader, LengthOfStayReader, PhenotypingReader, MultitaskReader

    # Set the paths to the data files
    processed_paths = {
        "IHM": Path(processedDir, "in-hospital-mortality"),
        "LOS": Path(processedDir, "length-of-stay"),
        "PHENO": Path(processedDir, "phenotyping"),
        "DECOMP": Path(processedDir, "decompensation"),
        "MULTI": Path(processedDir, "multitask")
    }

    discretized_paths = {
        "IHM": Path(discretizedDir, "in-hospital-mortality"),
        "LOS": Path(discretizedDir, "length-of-stay"),
        "PHENO": Path(discretizedDir, "phenotyping"),
        "DECOMP": Path(discretizedDir, "decompensation"),
        "MULTI": Path(discretizedDir, "multitask")
    }

    readers = {
        "IHM": InHospitalMortalityReader,
        "LOS": LengthOfStayReader,
        "PHENO": PhenotypingReader,
        "DECOMP": DecompensationReader,
        "MULTI": MultitaskReader
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
        if task in ["IHM", "PHENO", "MULTI"]:
            example_indices = list_file.index
        else:
            example_indices = list_file.groupby("stay")["period_length"].idxmax().values
        discretized_paths[task].mkdir(parents=True, exist_ok=True)
        list_file.loc[example_indices].to_csv(Path(discretized_paths[task], "listfile.csv"),
                                              index=False)

        print(f"{Fore.BLUE}Discretizing data for: {task} {Style.RESET_ALL}")

        with nullyaspin(f"") if os.getenv('GITHUB_ACTIONS') else \
             yaspin(color="green", text=f"") as sp:
            try:
                for impute_strategy in impute_strategies:
                    for start_time in start_times:
                        sp.text = f"Discretizing data with impute strategy '{impute_strategy}' and start mode '{start_time}'"
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
                                         columns=discretizer_header).to_csv(Path(
                                             target_dir, sample['name']),
                                                                            index=False)
                sp.ok("✅")
            except Exception as e:
                sp.fail("❌")
                import shutil
                shutil.rmtree(str(discretizedDir))
                raise e
