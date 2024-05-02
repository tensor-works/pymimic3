import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from metrics import Meter
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils import dict_subset
from functools import partial
from utils import update_json
from utils.IO import *


class StandardLogReg(object):

    def __init__(self,
                 task,
                 penalty="l2",
                 *,
                 dual=False,
                 tol=0.0001,
                 C=1,
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver="lbfgs",
                 max_iter=100,
                 multi_class="auto",
                 verbose=0,
                 warm_start=False,
                 n_jobs=None,
                 l1_ratio=None):
        """_summary_

        Args:
            task (_type_): _description_
            penalty (str, optional): _description_. Defaults to "l2".
            dual (bool, optional): _description_. Defaults to False.
            tol (float, optional): _description_. Defaults to 0.0001.
            C (int, optional): _description_. Defaults to 1.
            fit_intercept (bool, optional): _description_. Defaults to True.
            intercept_scaling (int, optional): _description_. Defaults to 1.
            class_weight (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            solver (str, optional): _description_. Defaults to "lbfgs".
            max_iter (int, optional): _description_. Defaults to 100.
            multi_class (str, optional): _description_. Defaults to "auto".
            verbose (int, optional): _description_. Defaults to 0.
            warm_start (bool, optional): _description_. Defaults to False.
            n_jobs (_type_, optional): _description_. Defaults to None.
            l1_ratio (_type_, optional): _description_. Defaults to None.
        """
        self._task = task
        self._possible_metrics = {
            "auc_roc": roc_auc_score,
            "auc_pr": lambda y_true, y_pred: auc(*precision_recall_curve(y_true, y_pred)[:2]),
            "auc_roc_micro": partial(roc_auc_score, average="micro"),
            "auc_roc_macro": partial(roc_auc_score, average="macro")
        }
        self._model = LogisticRegression(penalty,
                                         dual=dual,
                                         tol=tol,
                                         C=C,
                                         fit_intercept=fit_intercept,
                                         intercept_scaling=intercept_scaling,
                                         class_weight=class_weight,
                                         random_state=random_state,
                                         solver=solver,
                                         max_iter=max_iter,
                                         multi_class=multi_class,
                                         verbose=verbose,
                                         warm_start=warm_start,
                                         n_jobs=n_jobs,
                                         l1_ratio=l1_ratio)

    def fit(self, X, y):

        trainer_switch = {
            "PHENO": self._fit_phenotyping,
            "DECOMP": self._fit_dec_and_ihm,
            "IHM": self._fit_dec_and_ihm,
            "LOS": self._fit_los
        }
        return trainer_switch[self._task](X, y)

    def _fit_phenotyping(self, X, y):
        self._multi_target = MultiOutputClassifier(self._model, n_jobs=-1)
        return self._multi_target.fit(X, y)

    def _fit_dec_and_ihm(self, X, y):
        """"""
        return self._model.fit(X, y)  # super().fit(X, y)

    def _fit_los(self, X, y):
        """"""
        with Path(os.getenv("CONFIG"), "datasets.json") as config_dictionary:
            bins = config_dictionary['length_of_stay']['bins']

        y = np.digitize(y, bins)

        returns = self._model.fit(X, y)
        for key, value in self._model.__dict__:
            self.__dict__[key] = value
        return returns

    def __getattr__(self, name: str):
        """ Surrogate to the _model attributes internals.

        Args:
            name (str): name of the method/attribute

        Returns:
            any: method/attribute of _model
        """
        if name in ["fit", "_model", "evaluate"]:
            return self.__getattribute__(name)
        return getattr(self._model, name)

    def evaluate(self, X, y, metrics: list, storage_path: Path = None, set_name: str = 0):
        info_io(f"{set_name.capitalize()} metrics")
        if self._task == "PHENO":
            if not hasattr(self, "_multi_target"):
                self._multi_target = MultiOutputClassifier(self, n_jobs=-1)
            y_pred = self._multi_target.predict(X)
        else:
            y_pred = self._model.predict(X)
        metric_collection = dict()
        error_messages = list()
        for metric, func in dict_subset(self._possible_metrics, metrics).items():
            try:
                value = func(y_pred, y)
                print(f"{metric}: {value:.4f}")
                metric_collection.update({metric: value})
            except ValueError as error:
                error_messages.append(error)
        if storage_path:
            update_json(Path(storage_path, "history.json"), {set_name: metric_collection})
        for msg in error_messages:
            info_io("Exception has occured in model evaluation")
            info_io(msg)
