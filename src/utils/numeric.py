import torch
import numpy as np
from torcheval.metrics import metric


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
