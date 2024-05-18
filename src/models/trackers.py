import numpy as np
from pathlib import Path
from utils.IO import *
from storable import storable
from utils import update_json


@storable
class ModelHistory():

    train_loss: dict = {}
    val_loss: dict = {}
    best_val: dict = {"loss": float('inf'), "epoch": np.nan}
    best_train: dict = {"loss": float('inf'), "epoch": np.nan}
    test_loss: float = np.nan

    def to_json(self):
        update_json(Path(self._path.parent, f"{self._path.stem}.json"), self._progress)
        return self._progress


class LocalModelHistory():

    train_loss: dict = {}
    val_loss: dict = {}
    best_val: dict = {"loss": float('inf'), "epoch": np.nan}
    best_train: dict = {"loss": float('inf'), "epoch": np.nan}
    test_loss: float = np.nan

    def to_json(self):
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val": self.best_val,
            "best_train": self.best_train,
            "test_loss": self.test_loss
        }
