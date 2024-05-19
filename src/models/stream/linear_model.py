from pathlib import Path
from models.stream import AbstractRiverModel
from river.linear_model import ALMAClassifier as _ALMAClassifier
from river.linear_model import LogisticRegression as _LogisticRegression

__all__ = ["ALMAClassifier", "LogisticRegression"]


class ALMAClassifier(AbstractRiverModel, _ALMAClassifier):

    def __init__(self, model_path: Path, metrics: list, *args, **kwargs):
        self._default_name = "alma_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _ALMAClassifier.__init__(self, *args, **kwargs)


class LogisticRegression(AbstractRiverModel, _LogisticRegression):

    def __init__(self, model_path: Path, metrics: list, *args, **kwargs):
        self._default_name = "logistic_regression_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _LogisticRegression.__init__(self, *args, **kwargs)
