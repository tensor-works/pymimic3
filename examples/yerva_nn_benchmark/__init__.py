import argparse
import datasets
import pandas as pd
from pathlib import Path
from datasets.readers import SplitSetReader
from typing import List
from utils.IO import *
from examples.settings import *
from examples.example_utils import benchmark_split_reader, benchmark_split_subjects
from examples.yerva_nn_benchmark.scripts.logistic_regression import run_log_reg
from examples.yerva_nn_benchmark.scripts.lstm import run_standard_lstm
# from .decomp import logistic_regression, lstm_channel_wise, lstm, river_models
# from .ihm import logistic_regression, lstm_channel_wise, lstm, river_models
# from .los import logistic_regression, lstm_channel_wise, lstm, river_models
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creation of the original Yerva NN benchmark"
                                     "We also added some models for the fun of it :). "
                                     "Multitasking will be added during future maintenance.")
    parser.add_argument('--mimic_dir',
                        type=str,
                        default=EXAMPLE_DATA,
                        help="Path to your MIMIC-III dataset installation. "
                        "If not provided, the downloaded example dataset"
                        " will be used but the results will be invalid.")

    parser.add_argument('--tasks',
                        type=List[str],
                        default=["IHM", "LOS", "DECOMP", "PHENO"],
                        help="This is the list of tasks for which the benchmark is to be created. ")

    parser.add_argument(
        '--models',
        type=List[str],
        default=["logistic_regression", "lstm_channel_wise", "lstm", "river_models"],
        help="This is the list of models for which the benchmark is tó be created. ")

    args, _ = parser.parse_known_args()

    if not Path(YERVA_SPLIT, "testset.csv").is_file() or not Path(YERVA_SPLIT,
                                                                  "valset.csv").is_file():
        raise FileNotFoundError(
            f"The testset.csv or valset.csv file is missing. Please run the bash script "
            f"from examples/etc/setup.sh or .ps1"
            f"\nExpected location was {YERVA_SPLIT}")
    else:
        # These are fetched from the original github https://github.com/YerevaNN/mimic3-benchmarks
        test_subjects, val_subjects = benchmark_split_subjects()

    for task_name in args.tasks:
        info_io(f"Creating benchmark for task {task_name}")
        if not set(["lstm_channel_wise", "lstm"]) - set(args.models):
            reader = datasets.load_data(chunksize=75836,
                                        source_path=EXAMPLE_DATA_DEMO,
                                        storage_path=TEMP_DIR,
                                        discretize=True,
                                        time_step_size=1.0,
                                        start_at_zero=True,
                                        impute_strategy='previous',
                                        task=task_name)

            info_io(f"Splitting data for task {task_name}", level=0)

            split_reader = benchmark_split_reader(reader, test_subjects, val_subjects)

            if "lstm" in args.models:
                storage_path = Path(BENCHMARK_MODEL, task_name, "lstm")
                storage_path.mkdir(parents=True, exist_ok=True)

                run_standard_lstm(task_name=task_name,
                                  reader=split_reader,
                                  storage_path=storage_path,
                                  metrics=NETWORK_METRICS[task_name],
                                  params=STANDARD_LSTM_PARAMS[task_name])

        # Classical classifiers
        if False and not set(["logistic_regression", "river_models"]) - set(args.models):
            reader = datasets.load_data(chunksize=75836,
                                        source_path=EXAMPLE_DATA_DEMO,
                                        storage_path=TEMP_DIR,
                                        engineer=True,
                                        task=task_name)
            info_io(f"Splitting data for task {task_name}", level=0)

            split_reader = benchmark_split_reader(reader, test_subjects, val_subjects)

            if "logistic_regression" in args.models:
                storage_path = Path(BENCHMARK_MODEL, task_name, "logistic_regression")
                storage_path.mkdir(parents=True, exist_ok=True)

                run_log_reg(task_name=task_name,
                            reader=split_reader,
                            storage_path=storage_path,
                            metrics=LOG_METRICS[task_name],
                            params=LOG_REG_PARAMS[task_name])
            pass
