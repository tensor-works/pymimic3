import subprocess
import warnings
import sys
import os
import re
import runpy
import pandas as pd
from pathlib import Path
from colorama import Fore, Style


def check_required_files(extracted_directory: Path):
    subject_directories = [
        subj_dir for subj_dir in extracted_directory.iterdir() if subj_dir.is_dir()
    ]
    if not subject_directories:
        raise FileNotFoundError(f"No subject directories found in {extracted_directory}")

    for subj_dir in subject_directories:
        if not subj_dir.is_dir():
            continue

        # Define the patterns to match required files
        required_patterns = [
            r"diagnoses\.csv",
            r"episode\d+_timeseries\.csv",
            r"episode\d+\.csv",
            r"events\.csv",
            r"stays\.csv",
        ]

        # Check files in the directory
        files = [f.name for f in subj_dir.iterdir() if f.is_file()]
        matches = {pattern: False for pattern in required_patterns}

        # Check each file against each pattern
        for file in files:
            for pattern in required_patterns:
                if re.search(pattern, file):
                    matches[pattern] = True

        # Check if all patterns were matched
        all_matched = all(matches.values())
        if not all_matched:
            raise FileNotFoundError(f"Missing required files in {subj_dir}: {matches}")


if __name__ == "__main__":
    csvDir = Path(sys.argv[1])
    extractedDir = Path(sys.argv[2])
    repositoryDir = Path(sys.argv[3])

    sys.path.append(str(repositoryDir))

    if not csvDir.exists():
        raise FileNotFoundError(f"MIMIC-DEMO directory not found at location: {csvDir}")
    if not extractedDir.exists():
        extractedDir.mkdir(parents=True, exist_ok=True)

    # Define the tasks and corresponding modules
    tasks = [
        ("mimic3benchmark.scripts.extract_subjects", (str(csvDir), str(extractedDir))),
        ("mimic3benchmark.scripts.validate_events", (str(extractedDir),)),
        ("mimic3benchmark.scripts.extract_episodes_from_subjects", (str(extractedDir),)),
    ]
    try:
        for module, args in tasks:
            print(f"{Fore.BLUE}Running {module} with args: {*args,}{Style.RESET_ALL}")
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

            # Modify sys.argv to simulate command-line arguments then run
            original_argv = sys.argv.copy()
            sys.argv = [module] + list(args)
            runpy.run_module(module, run_name="__main__")
            sys.argv = original_argv

        check_required_files(extractedDir)
    except Exception as e:
        print(f"{Fore.RED}Error running benchmarks:{Style.RESET_ALL} {e}")
        import shutil
        shutil.rmtree(extractedDir)
        raise e
