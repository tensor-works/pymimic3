from torcheval.metrics import BinaryAUPRC, MulticlassAUPRC, MultilabelAUPRC
from copy import deepcopy

# TODO! This absolutetly needs testing


class AUCPRC(object):

    def __init__(self, task: str, num_classes: int = 1):
        if task == "binary":
            self.metric = BinaryAUPRC()
        elif task == "multiclass":
            self.metric = MulticlassAUPRC(num_classes=num_classes)
        elif task == "multilabel":
            self.metric = MultilabelAUPRC(num_labels=num_classes)
        else:
            raise ValueError("Unsupported task type or activation function")
        self._task = task

    def update(self, predictions, labels):
        # Reshape predictions and labels to handle the batch dimension
        if self._task == "binary":
            predictions = predictions.view(-1)
            labels = labels.view(-1)
        else:
            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1)

        self.metric.update(predictions, labels)

    def to(self, device):
        # Move the metric to the specified device
        self.metric = self.metric.to(device)
        return self

    def __getattr__(self, name):
        # Redirect attribute access to self.metric if it exists there
        if hasattr(self.metric, name) and not name in ["update", "to", "__dict__"]:
            return getattr(self.metric, name)
        # if name in self.__dict__:
        #     return self.__dict__[name]
        # raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ['_task', 'metric']:
            # Set attributes normally if they are part of the AUCPRC class
            super().__setattr__(name, value)
        elif hasattr(self, 'metric') and hasattr(self.metric, name):
            # Redirect attribute setting to self.metric if it exists there
            setattr(self.metric, name, value)
        else:
            # Set attributes normally otherwise
            super().__setattr__(name, value)

    def __deepcopy__(self, memo):
        # Create a new instance of the class
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy the _task attribute
        setattr(result, '_task', deepcopy(self._task, memo))

        # Deep copy the metric attribute
        if hasattr(self, 'metric'):
            setattr(result, 'metric', deepcopy(self.metric, memo))

        return result
