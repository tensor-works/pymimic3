from pathlib import Path
from models.stream import AbstractRiverModel
from river.forest import ARFClassifier as _ARFClassifier
from river.forest import AMFClassifier as _AMFClassifier

__all__ = ["ARFClassifier", "AMFClassifier"]


class ARFClassifier(AbstractRiverModel, _ARFClassifier):

    def __init__(self, model_path: Path = None, metrics: list = [], *args, **kwargs):
        self._default_name = "arf_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _ARFClassifier.__init__(self, *args, **kwargs)


class AMFClassifier(AbstractRiverModel, _AMFClassifier):

    def __init__(self, model_path: Path = None, metrics: list = [], *args, **kwargs):
        self._default_name = "amf_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _AMFClassifier.__init__(self, *args, **kwargs)
