import os
import gdown
import subprocess
from yaspin import yaspin
from pathlib import Path

# Google drive location
google_id = "1drPlXCTGPijJHoDb8v7_VJJaUUyvDu56"  # TODO! relocate

# Create the data folder
data_folder = Path(os.getenv("TESTS"), "data")
data_folder.mkdir(parents=True, exist_ok=True)

# Download the control dataset
save_file_path = Path(data_folder, "control-dataset.tar.xz")
url = f'https://drive.google.com/uc?id={google_id}'

with yaspin(color="green", text="Downloading the control dataset") as sp:
    try:
        gdown.download(url=str(url), output=str(save_file_path), quiet=False)
        sp.text = "Download complete"
        sp.ok("✅")
    except Exception as e:
        sp.fail("❌")
        raise e

# Unzip the file
try:
    # Use tar command to extract the file
    subprocess.run(["tar", "-xf", str(save_file_path), "-C", str(data_folder)], check=True)
    print(f"All files have been extracted to {data_folder}")
    # Remove the tar file
    save_file_path.unlink()
except subprocess.CalledProcessError as e:
    print(f"An error occurred while extracting the file: {e}")
