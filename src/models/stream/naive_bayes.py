from pathlib import Path
from models.stream import AbstractRiverModel
from river.naive_bayes import GaussianNB as _GaussianNB

__all__ = ["GaussianNBClassifier"]


class GaussianNBClassifier(AbstractRiverModel, _GaussianNB):

    def __init__(self, model_path: Path, metrics: list, *args, **kwargs):
        self._default_name = "gaussian_nb_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _GaussianNB.__init__(self, *args, **kwargs)
