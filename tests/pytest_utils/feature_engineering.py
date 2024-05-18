import re
import pandas as pd
import numpy as np


def extract_test_ids(df: pd.DataFrame):
    regex = r"(\d+)_episode(\d+)_timeseries\.csv"
    df["SUBJECT_ID"] = df["0"].apply(lambda x: re.search(regex, x).group(1))
    df["ICUSTAY_ID"] = df["0"].apply(lambda x: re.search(regex, x).group(2))
    df = df.drop("0", axis=1)
    return df


def concatenate_dataset(data) -> pd.DataFrame:
    data_stack = list()
    for subject_id, subject_stays in data.items():
        for stay_id, frame in subject_stays.items():
            if len(np.squeeze(frame).shape) == 1:
                data_stack.append(np.squeeze(frame).tolist() + [subject_id, stay_id])
            else:
                for row in np.squeeze(frame).tolist():
                    data_stack.append(row + [subject_id, stay_id])

    dfs = pd.DataFrame(data_stack,
                       columns=[str(idx) for idx in range(1, 715)] + ["SUBJECT_ID", "ICUSTAY_ID"])

    if not len(dfs):
        return
    return dfs
