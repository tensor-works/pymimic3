import os
import subprocess


def run_test_scripts(directory):
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file starts with 'test_' and ends with '.py'
            if file.startswith('test_') and file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Running {file_path}...")
                try:
                    # Run the script
                    subprocess.run(['python', file_path], check=True)
                    print(f"Finished running {file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running {file_path}: {e}")


if __name__ == "__main__":
    # Specify the directory to search in
    root_directory = os.getcwd()  # Use current working directory as the root
    run_test_scripts(root_directory)
