import pandas as pd
import torch
import numpy as np


class RunningAvg():

    def __init__(self):
        self._average = np.nan
        self.reset()

    def reset(self):
        self._count = 0
        self._average = 0.0

    def update(self, loss: torch.Tensor):
        self._count += 1
        self._average += (loss - self._average) / self._count

    def compute(self) -> float:
        return self._average

    def __repr__(self):
        return f"RunningAvg(count={self._count}, average={self.compute():.4f})"


if __name__ == "__main__":
    ...


def is_numerical(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame contains only numerical data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    bool
        True if the DataFrame is numerical, False otherwise.
    """
    # This is the worst implementation but what works works
    try:
        df.astype(float)
        return True
    except:
        pass
    if (df.dtypes == object).any() or\
       (df.dtypes == "category").any() or\
       (df.dtypes == "datetime64[ns]").any():
        return False
    return True
