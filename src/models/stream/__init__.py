import os
import pickle
from typing import Dict
from river.metrics.base import Metric
from pathlib import Path
from models.stream.mappings import metric_mapping
from generators.stream import RiverGenerator
from models.trackers import RiverHistory, LocalRiverHistory
from tensorflow.keras.utils import Progbar
from utils import to_snake_case, dict_subset
from abc import ABC
from copy import deepcopy


class AbstractRiverModel(ABC):

    def __init__(self, model_path: Path = None, metrics: list = [], name: str = None):
        self._name = (name if name is not None else self._default_name)
        self._model_path = model_path

        if self._model_path is not None:
            # Persistent history
            self._model_path.mkdir(parents=True, exist_ok=True)
            self._history = RiverHistory(Path(self._model_path, "history"))
        else:
            # Mimics the storable
            self._history = LocalRiverHistory()
        if not self.load():
            self._metrics = self._init_metrics(metrics)
            self._train_metrics = self._init_metrics(metrics)
            self._test_metrics = self._init_metrics(metrics, prefix="test")
            self._val_metrics = self._init_metrics(metrics, prefix="val")

    def _init_metrics(self, metrics, prefix: str = None) -> Dict[str, Metric]:
        return_metrics = dict()
        for metric in metrics:
            if isinstance(metric, str):
                metric_name = metric
                metric = metric_mapping[metric]()
            else:
                metric_name = to_snake_case(metric.__class__.__name__)
            if prefix is not None:
                metric_name = f"{prefix}_{metric_name}"
            return_metrics[metric_name] = metric
        return return_metrics

    def fit(self,
            train_generator: RiverGenerator,
            val_generator: RiverGenerator = None,
            model_path: Path = None):
        if model_path is not None:
            self._model_path = model_path
        self.load()
        self.train(generator=train_generator, has_val=val_generator is not None)
        if val_generator is not None:
            self.evaluate(generator=val_generator, is_val=True)

        self.save()

    def train(self, generator: RiverGenerator, has_val: bool = False):
        generator_size = len(generator)
        self._train_progbar = Progbar(generator_size)

        for batch_idx, (x, y) in enumerate(generator):
            y_pred = self.predict_one(x)
            for name, metric in self._train_metrics.items():
                metric.update(y, y_pred)
            self.learn_one(x, y)

            self._train_progbar.update(batch_idx + 1,
                                       values=self._get_metrics(self._train_metrics),
                                       finalize=(batch_idx == generator_size and not has_val))
        for name, metric in self._train_metrics.items():
            self._history.train_metrics[name] = metric.get()

        return self._train_metrics

    def _get_metrics(self, metric):
        keys = list(metric.keys())
        values = [value.get() for value in metric.values()]
        return tuple(zip(keys, values))

    def predict(self):
        pass

    def evaluate(self, generator: RiverGenerator, is_val: bool = False, is_test: bool = False):
        if is_val:
            eval_metric = self._val_metrics
        elif is_test:
            eval_metric = self._test_metrics
        else:
            eval_metric = deepcopy(self._metrics)

        for _, (x, y) in enumerate(generator):
            y_pred = self.predict_one(x)
            for name, metric in eval_metric.items():
                metric.update(y, y_pred)
        if is_val:
            self._train_progbar.update(self._train_progbar.target,
                                       values=self._get_metrics(self._train_metrics) +
                                       self._get_metrics(self._val_metrics))
            for name, metric in self._val_metrics.items():
                self._history.val_metrics[name] = metric.get()

        return eval_metric

    def save(self, model_path=None):
        """_summary_
        """
        if model_path is not None:
            storage_path = Path(model_path, f"{self._name}.pkl")
        elif self._model_path is not None and self._name is not None:
            storage_path = Path(self._model_path, f"{self._name}.pkl")
        if storage_path is not None:
            with open(storage_path, "wb") as save_file:
                # Storables can't be pickled
                save_dict = dict_subset(
                    self.__dict__,
                    list(set(self.__dict__.keys()) - set(["_history", "_train_progbar"])))
                pickle.dump(obj=save_dict, file=save_file, protocol=2)

    def load(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        def inner_load(self, path: Path):
            if path.is_file():
                if os.path.getsize(path) > 0:
                    with open(path, "rb") as load_file:
                        load_params = pickle.load(load_file)
                    for key, value in load_params.items():
                        setattr(self, key, value)
                    return 1
            return 0

        if self._model_path is not None and self._name is not None:
            return inner_load(self, Path(self._model_path, f"{self._name}.pkl"))

        return 0
