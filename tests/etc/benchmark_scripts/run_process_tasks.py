import subprocess
import warnings
import sys
import runpy
import pandas as pd
from pathlib import Path
from colorama import Fore, Style


def run_script(module_name, input_dir, output_dir):
    warnings.filterwarnings("ignore",
                            category=pd.core.common.PerformanceWarning)  # Ignore PerformanceWarning
    command = [sys.executable, '-m', module_name, input_dir, output_dir]
    subprocess.run(command)


if __name__ == "__main__":
    extractedDir = Path(sys.argv[1])
    processedDir = Path(sys.argv[2])
    repositoryDir = Path(sys.argv[3])

    sys.path.append(str(repositoryDir))

    # Define the tasks and corresponding modules
    task_names = [
        "in-hospital-mortality", "decompensation", "length-of-stay", "phenotyping", "multitask"
    ]

    # Run each task
    for task in task_names:
        try:
            module = f"mimic3benchmark.scripts.create_{task.replace('-', '_')}"
            task_target_dir = Path(processedDir, task)
            args = (str(extractedDir), str(task_target_dir))
            print(f"{Fore.BLUE}Running {module} with args: {*args,}{Style.RESET_ALL}")

            # Modify sys.argv to simulate command-line arguments then run
            original_argv = sys.argv.copy()
            sys.argv = [module] + list(args)
            runpy.run_module(module, run_name="__main__")
            sys.argv = original_argv

        except Exception as e:
            import shutil

            print(f"{Fore.RED}Error running benchmarks:{Style.RESET_ALL} {e}")
            raise e
