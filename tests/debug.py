import os
import subprocess
from utils.jsons import load_json
from pathlib import Path
from tests.tsettings import *
from utils.jsons import write_json

test_history_file = Path(os.getenv("TESTS"), "test_history.json")


def run_test_scripts(directory: str, test_history: list):
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file starts with 'test_' and ends with '.py'
            if file.startswith('test_') and file.endswith('.py') and not file in test_history:
                file_path = os.path.join(root, file)
                print(f"Running {file_path}...")
                # Run the script
                subprocess.run(['python', file_path], check=True)
                print(f"Finished running {file_path}")
                test_history.append(file)
                write_json(test_history_file, test_history)


if __name__ == "__main__":
    # Specify the directory to search in
    root_directory = os.getcwd()  # Use current working directory as the root
    if test_history_file.is_file():
        test_history = load_json(test_history_file)
    else:
        test_history = list()
    run_test_scripts(root_directory, test_history)
