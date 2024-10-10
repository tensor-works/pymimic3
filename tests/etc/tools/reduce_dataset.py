# This was used to reduce the dataset to speed up CICD
import os
import shutil
from pathlib import Path

# no mortality
no_mortality = [
    10017,
    20124,
    42458,
    10117,
    40601,
    41976,
    42321,
    10088,
    10119,
    41914,
]

# mortality
mortality = [
    10069,
    10019,
    10011,
    43909,
    10102,
    44154,
    10036,
    40503,
    10112,
    10111,
]

# Directory to check
directory = Path(os.getenv("TESTS"), "data", "control-dataset", "extracted")

# Convert lists to a set for faster lookup
safe_dirs = set(no_mortality + mortality)

# Traverse the directory
for folder in Path(directory).iterdir():
    folder_path = Path(directory, folder)
    # Check if the folder name is a digit and not in the safe list
    if str(folder).isdigit() and int(folder) not in safe_dirs:
        # Remove the folder if it is not in the safe lists
        shutil.rmtree(str(folder_path))
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Skipped folder: {folder_path}")

# Load the data
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(directory, 'all_diagnoses.csv'))

safe_subject_ids = set(no_mortality + mortality)
filtered_df = df[df['SUBJECT_ID'].isin(safe_subject_ids)]
filtered_df.to_csv(Path(directory, 'all_diagnoses.csv'), index=False)

print("Filtered CSV saved as 'filtered_diagnoses.csv'.")

# Load the data from both CSV files
all_stays = pd.read_csv(Path(directory, 'all_stays.csv'))
phenotype_label = pd.read_csv(Path(directory, 'phenotype_labels.csv'), header=None)

filtered_stays = all_stays[all_stays['SUBJECT_ID'].isin(safe_dirs)]
filtered_labels = phenotype_label.iloc[filtered_stays.index]

# Save the filtered data back to CSV files
filtered_stays.to_csv(Path(directory, 'all_stays.csv'), index=False)
filtered_labels.to_csv(Path(directory, 'phenotype_labels.csv'), index=False, header=None)

print("Filtered CSV files saved as 'filtered_all_stays.csv' and 'filtered_phenotype_label.csv'.")
