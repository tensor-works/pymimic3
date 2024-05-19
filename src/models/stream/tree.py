from pathlib import Path
from models.stream import AbstractRiverModel
from river.tree import HoeffdingAdaptiveTreeClassifier as _HoeffdingAdaptiveTreeClassifier
from river.tree import HoeffdingTreeClassifier as _HoeffdingTreeClassifier

__all__ = ["HoeffdingAdaptiveTreeClassifier", "HoeffdingTreeClassifier"]


class HoeffdingAdaptiveTreeClassifier(AbstractRiverModel, _HoeffdingAdaptiveTreeClassifier):

    def __init__(self, model_path: Path, metrics: list, *args, **kwargs):
        self._default_name = "hoeffding_adaptive_tree_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _HoeffdingAdaptiveTreeClassifier.__init__(self, *args, **kwargs)


class HoeffdingTreeClassifier(AbstractRiverModel, _HoeffdingTreeClassifier):

    def __init__(self, model_path: Path, metrics: list, *args, **kwargs):
        self._default_name = "hoeffding_tree_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _HoeffdingTreeClassifier.__init__(self, *args, **kwargs)
