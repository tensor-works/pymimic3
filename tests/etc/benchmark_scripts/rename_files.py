import pdb
import re
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from yaspin import yaspin
from pathlib import Path

load_dotenv()

if __name__ == "__main__":
    extractedDir = Path(sys.argv[1])
    # I want to rename the files episodeX.csv and episodeX_timeseries.csv by replacing X, which in this case is a random number by the icustay
    with yaspin(color="green", text="Converting file names") as sp:
        try:
            for subject_dir in extractedDir.iterdir():
                sp.text = f"Converting file names in {subject_dir.name}"
                if not subject_dir.is_dir():
                    continue
                for csv in subject_dir.iterdir():
                    if "episode" in csv.name and not "timeseries" in csv.name:
                        df = pd.read_csv(csv, na_values=[''], keep_default_na=False)
                        icustay_id = df["Icustay"].iloc[0]
                        file_index = re.findall(r'\d+', csv.name).pop()
                        csv.rename(Path(csv.parent, f"episode{icustay_id}.csv"))
                        Path(csv.parent, f"episode{file_index}_timeseries.csv").rename(
                            Path(csv.parent, f"episode{icustay_id}_timeseries.csv"))
            sp.text = "Done converting file names"
            sp.ok("✅")
        except Exception as e:
            sp.fail("❌")
            raise e
