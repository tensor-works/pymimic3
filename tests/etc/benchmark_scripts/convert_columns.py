import sys
import os
import pandas as pd
from pathlib import Path
from yaspin import yaspin
from tests.pytest_utils.reporting import nullyaspin
from datasets.mimic_utils import upper_case_column_names

if __name__ == "__main__":
    demoDataDir = Path(sys.argv[1])
    with nullyaspin(f"Converting columns to uppercase...") if os.getenv('GITHUB_ACTIONS') else \
         yaspin(color="green", text=f"Converting columns to uppercase...") as sp:
        try:
            for csv in demoDataDir.iterdir():
                sp.text = f"Upper casing columns in {csv.name}"
                if csv.is_dir() or csv.suffix != ".csv":
                    continue

                df = pd.read_csv(csv,
                                 dtype={
                                     "ROW_ID": 'Int64',
                                     "ICUSTAY_ID": 'Int64',
                                     "HADM_ID": 'Int64',
                                     "SUBJECT_ID": 'Int64',
                                     "row_id": 'Int64',
                                     "icustay_id": 'Int64',
                                     "hadm_id": 'Int64',
                                     "subject_id": 'Int64'
                                 },
                                 na_values=[''],
                                 keep_default_na=False,
                                 low_memory=False)
                df = upper_case_column_names(df)
                df.to_csv(csv, index=False)
            sp.text = "Done formating columns"
            sp.ok("✅")
        except Exception as e:
            sp.fail("❌")
            raise e
