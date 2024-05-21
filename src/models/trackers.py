import numpy as np
from pathlib import Path
from utils.IO import *
from storable import storable
from utils import update_json
from settings import *


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


@storable
class RiverHistory():

    train_metrics: dict = {}
    val_metrics: dict = {}
    test_metrics: dict = {}

    def _allowed_key(self, key: str):
        return not any([key.endswith(metric) for metric in TEXT_METRICS])

    def to_text(self, report_path: Path, report_dict: dict):
        report = ""
        for key, value in report_dict.items():
            report += f"{key}:\n {value}\n"

        with open(report_path, 'w') as file:
            file.write(report)
        return report

    def to_json(self):
        numeric_dict = {}
        for attribute, history in self._progress.items():
            numeric_dict[attribute] = dict()
            for metric, value in history.items():
                if self._allowed_key(metric):
                    numeric_dict[attribute][metric] = value
        report_dict = {}
        for attribute, history in self._progress.items():
            for metric, value in history.items():
                if not metric in numeric_dict[attribute]:
                    report_dict[metric] = value

        update_json(Path(self._path.parent, f"{self._path.stem}.json"), numeric_dict)
        self.to_text(Path(self._path.parent, f"{self._path.stem}.txt"), report_dict)
        return self._progress


class LocalRiverHistory():

    train_metrics: dict = {}
    val_metrics: dict = {}
    test_metrics: dict = {}

    def _allowed_key(self, key: str):
        return not any([key.endswith(metric) for metric in self._non_numeric_metrics])

    def to_json(self):
        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics
        }
