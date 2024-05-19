from pathlib import Path
from models.stream import AbstractRiverModel
from river.neighbors import KNNClassifier as _KNNClassifier

__all__ = ["KNNClassifier"]


class KNNClassifier(AbstractRiverModel, _KNNClassifier):

    def __init__(self, model_path: Path, metrics: list, *args, **kwargs):
        self._default_name = "knn_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _KNNClassifier.__init__(self, *args, **kwargs)
