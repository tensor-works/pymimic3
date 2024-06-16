import os
import collections
import copy
import functools
import pickle
import warnings
from typing import Dict
from river.metrics.base import Metric
from river import base
from river import linear_model
from pathlib import Path
from models.stream.mappings import metric_mapping
from generators.stream import RiverGenerator
from models.trackers import RiverHistory, LocalRiverHistory
from keras.utils import Progbar
from utils import to_snake_case, dict_subset
from abc import ABC
from copy import deepcopy
from settings import *


class AbstractRiverModel(ABC):

    def __init__(self, model_path: Path = None, metrics: list = [], name: str = None):
        self._name = (name if name is not None else self._default_name)
        self._model_path = model_path

        self._history = self._init_history(path=model_path)

        if not self.load():
            self._metrics = self._init_metrics(metrics)
            self._train_metrics = self._init_metrics(metrics)
            self._test_metrics = self._init_metrics(metrics, prefix="test")
            self._val_metrics = self._init_metrics(metrics, prefix="val")

    def _init_history(self, path: Path):
        if self._model_path is not None:
            # Persistent history
            self._model_path.mkdir(parents=True, exist_ok=True)
            return RiverHistory(Path(path, "history"))
        else:
            # Mimics the storable
            return LocalRiverHistory()

    def _init_metrics(self, metrics, prefix: str = None) -> Dict[str, Metric]:
        return_metrics = dict()
        for metric in metrics:
            if isinstance(metric, str):
                metric_name = metric
                metric = metric_mapping[metric]
            else:
                metric_name = to_snake_case(metric.__name__)
            if isinstance(metric, type):
                metric = metric()
            if prefix is not None:
                metric_name = f"{prefix}_{metric_name}"
            return_metrics[metric_name] = metric
        return return_metrics

    def _update_metrics(self, metrics: Dict[str, Metric], x, y_true, y_pred):
        y_label = None

        for _, metric in metrics.items():
            if not hasattr(metric, "requires_labels") or not metric.requires_labels:
                metric.update(y_true, y_pred)
            else:
                if y_label is None:
                    y_label = self.predict_one(x)
                metric.update(y_true, y_label)

    def _update_history(self, history_dict: dict, metrics: Dict[str, Metric]):
        for name, metric in metrics.items():
            try:
                history_dict[name] = metric.get()
            except NotImplementedError as e:
                history_dict[name] = metric

    def _allowed_key(self, key: str):
        return not any([key.endswith(metric) for metric in TEXT_METRICS])

    def fit(self,
            train_generator: RiverGenerator,
            val_generator: RiverGenerator = None,
            model_path: Path = None):
        if model_path is not None:
            self._model_path = model_path
            self._history = self._init_history(path=model_path)
        self.load()
        self.train(generator=train_generator, has_val=val_generator is not None)
        if val_generator is not None:
            self.evaluate(generator=val_generator, is_val=True)

        self.save()
        return self._history.to_json()

    def train(self, generator: RiverGenerator, has_val: bool = False):
        generator_size = len(generator)
        self._train_progbar = Progbar(generator_size)

        for batch_idx, (x, y) in enumerate(generator):
            self.learn_one(x, y)
            if hasattr(self, "predict_proba_one"):
                y_pred = self.predict_proba_one(x)
            else:
                y_pred = self.predict_one(x)
            self._update_metrics(metrics=self._train_metrics, x=x, y_true=y, y_pred=y_pred)
            self._train_progbar.update(batch_idx + 1,
                                       values=self._get_metrics(self._train_metrics),
                                       finalize=(batch_idx == generator_size and not has_val))

        self._update_history(history_dict=self._history.train_metrics, metrics=self._train_metrics)
        return self._train_metrics

    def _get_metrics(self, metrics: Dict[str, Metric]):
        keys = list([metric for metric in metrics.keys() if self._allowed_key(metric)])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            values = [metrics[key].get() for key in keys]
        return tuple(zip(keys, values))

    def predict(self):
        pass

    def test(self, generator: RiverGenerator):
        metrics = self.evaluate(generator=generator, is_test=True)
        self._history.to_json()
        return metrics

    def evaluate(self, generator: RiverGenerator, is_val: bool = False, is_test: bool = False):
        if is_val:
            eval_metric = self._val_metrics
        elif is_test:
            eval_metric = self._test_metrics
        else:
            eval_metric = deepcopy(self._metrics)

        for _, (x, y) in enumerate(generator):
            y_pred = self.predict_one(x)
            self._update_metrics(metrics=eval_metric, x=x, y_true=y, y_pred=y_pred)

        if is_val:
            self._train_progbar.update(self._train_progbar.target,
                                       values=self._get_metrics(self._train_metrics) +
                                       self._get_metrics(self._val_metrics))

            self._update_history(history_dict=self._history.val_metrics, metrics=self._val_metrics)

        if is_test:
            self._update_history(history_dict=self._history.test_metrics,
                                 metrics=self._test_metrics)

        return eval_metric

    def save(self, model_path=None):
        """_summary_
        """
        if model_path is not None:
            storage_path = Path(model_path, f"{self._name}.pkl")
        elif self._model_path is not None and self._name is not None:
            storage_path = Path(self._model_path, f"{self._name}.pkl")
        else:
            storage_path = None
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


class AbstractMultioutputClassifier(base.Wrapper, base.Classifier):

    def __init__(self, classifier):
        self.classifier = classifier
        new_clf = functools.partial(copy.deepcopy, classifier)
        self.classifiers = collections.defaultdict(new_clf)
        self.classes = set()

    @property
    def _wrapped_model(self):
        return self.classifier

    @property
    def _multiclass(self):
        return True

    @classmethod
    def _unit_test_params(cls):
        yield {"classifier": linear_model.LogisticRegression()}

    def learn_one(self, x, y, **kwargs):
        for label_name, y_label in y.items():
            self.classes.add(label_name)
            self.classifiers[label_name].learn_one(x, y_label)

    def predict_one(self, x, **kwargs):
        predictions = {}
        for label in self.classifiers:
            predictions[label] = self.classifiers[label].predict_one(x)

        return predictions

    def predict_proba_one(self, x, predict_for=None):
        predictions = {}
        for label in self.classifiers:
            predictions[label] = self.classifiers[label].predict_proba_one(x)
            if predict_for is not None:
                predictions[label] = predictions[label][predict_for]

        return predictions
