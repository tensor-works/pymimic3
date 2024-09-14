import torch
from typing import Type, Literal, Optional, Union, List, Any
from torcheval.metrics import BinaryAUPRC as _BinaryAUPRC
from torcheval.metrics import MulticlassAUPRC as _MulticlassAUPRC
from torcheval.metrics import MultilabelAUPRC
from torcheval.metrics import metric
from torchmetrics import AUROC as _AUROC
from torchmetrics import MeanAbsoluteError
from copy import deepcopy
from metrics import CustomBins, LogBins


class BinedMAE(MeanAbsoluteError):
    """
    Compute the Mean Absolute Error (MAE) for length of stay (LOS) from binned predictions and targets.
    
    The function computes the MAE between the bin means of the predictions and targets. The inputs and
    targets can be either one-hot encoded or ordinal, and the MAE is calculated based on either custom or
    logarithmic (log) binning methods.
    
    Parameters
    ----------
    bining : str
        The binning method to use. Must be one of {'log', 'custom'}.
    
    Returns
    -------
    float
        The computed MAE as a numeric value.
    
    Notes
    -----
    The MAE is computed between the bin means of the inputs and targets. The specific binning method
    ('log' or 'custom') is applied to calculate the means before computing the error.
    """

    def __init__(self, bining: Literal["log", "custom"], *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._binning = bining
        if self._binning == "custom":
            self._means = torch.tensor(CustomBins.means, dtype=torch.float32)
        elif self._binning == "log":
            self._means = torch.tensor(LogBins.means, dtype=torch.float32)
        else:
            raise ValueError(f"Binning must be one of 'log' or 'custom' but is {bining}.")

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        # Input is one-hot
        if input.dim() > 1:
            input = torch.argmax(input, axis=-1)

        input_means = self._means[input]

        # Target is one-hot
        if target.dim() > 1:
            target = torch.argmax(target, axis=-1)

        target_means = self._means[target]
        super().update(input_means, target_means)

    def to(self, *args, **kwargs):
        self._means = self._means.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class BinaryAUPRC(_BinaryAUPRC):

    def update(self, input: torch.Tensor, target: torch.Tensor):
        # Reshape predictions and labels to handle the batch dimension
        input = input.view(-1)
        target = target.view(-1)
        super().update(input, target)


class MulticlassAUPRC(_MulticlassAUPRC):

    def update(self, input: torch.Tensor, target: torch.Tensor):
        # Reshape predictions and labels to handle the batch dimension
        target = target.view(-1)
        super().update(input, target)


class AUPRC(metric.Metric[torch.Tensor]):
    """
    A metric class that wraps the Area Under Precision-Recall Curve (AUPRC) from `torcheval`.

    This class provides a way to instantiate the AUPRC metric with a task type and the required number of labels or classes.
    It behaves similarly to the metric implementations in the base torch packages.

    Parameters
    ----------
    task : {'binary', 'multiclass', 'multilabel'}
        The type of task for which the AUPRC metric is being calculated.
    num_labels : int, optional
        The number of labels for multilabel tasks. Defaults to 1.
    num_classes : int, optional
        The number of classes for multiclass tasks. Defaults to 1.
    average : {'macro', 'weighted', 'none', 'micro'}, optional
        The method to average the precision-recall scores across different classes. Defaults to 'macro'.

    Notes
    -----
    This class provides a wrapper around the `torcheval` implementation of AUPRC, allowing the metric to be easily used in 
    binary, multiclass, or multilabel tasks with flexible options for handling multiple classes or labels.
    """

    def __new__(cls,
                task: Literal["binary", "multiclass", "multilabel"],
                num_labels: int = 1,
                num_classes: int = 1,
                average: Literal["macro", "weighted", "none", "micro"] = "macro"):

        if average not in ["macro", "micro", "none"]:
            raise ValueError("Average must be one of 'macro', 'micro', or 'none'"
                             f" but is {average}")

        if task == "binary" or average == "micro":
            metric = BinaryAUPRC()
        elif task == "multiclass":
            # Some debate in the net but in torch this is one-vs-all
            metric = MulticlassAUPRC(num_classes=num_classes, average=average)
        elif task == "multilabel":
            # This is multiple positives allowed
            metric = MultilabelAUPRC(num_labels=num_labels, average=average)
        else:
            raise ValueError("Unsupported task type or activation function")

        metric._task = task
        metric._average = average

        return metric


class AUROC(_AUROC):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUROC) metric.

    This class is a subclass of `_AUROC` and extends it to handle binary, multiclass, and multilabel tasks. 
    It supports thresholding, averaging methods, and options for specifying the number of labels or classes.

    Parameters
    ----------
    task : {'binary', 'multiclass', 'multilabel'}
        The type of task for which the AUROC is being computed.
    thresholds : int or list of float or torch.Tensor, optional
        The number of thresholds or specific threshold values to compute AUROC. If `None`, the metric will be calculated without thresholds.
    num_labels : int, optional
        The number of labels in the multilabel case. Required if `task="multilabel"`.
    num_classes : int, optional
        The number of classes in the multiclass case. Required if `task="multiclass"`.
    average : {'macro', 'weighted', 'none', 'micro'}, optional
        The averaging method for the AUROC across multiple classes or labels. Defaults to 'macro'.
    max_fpr : float, optional
        The maximum false positive rate. If specified, only the AUROC up to this value is computed. Defaults to `None`.
    ignore_index : int, optional
        Specifies a target value that should be ignored. Defaults to `None`.
    validate_args : bool, optional
        If `True`, validates the inputs and arguments for correctness. Defaults to `True`.
    """

    def __new__(
        cls: Type["_AUROC"],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, List[float], torch.Tensor]] = None,
        num_labels: Optional[int] = None,
        num_classes: Optional[int] = None,
        average: Optional[Literal["macro", "weighted", "none", "micro"]] = "macro",
        max_fpr: Optional[float] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
    ):

        if average == "micro" and task == "multilabel":
            task = "binary"

        kwargs = {}

        if num_classes is not None:
            kwargs["num_classes"] = num_classes

        if num_labels is not None:
            kwargs["num_labels"] = num_labels

        metric = super().__new__(cls,
                                 task=task,
                                 thresholds=thresholds,
                                 average="none" if average == "micro" else average,
                                 max_fpr=max_fpr,
                                 ignore_index=ignore_index,
                                 validate_args=validate_args,
                                 **kwargs)
        metric._average = average
        return metric

    # You might want to override update and compute methods if needed
    def update(self,
               input: torch.Tensor,
               target: torch.Tensor,
               weight: torch.Tensor = None,
               *args,
               **kwargs) -> None:

        if self._average == "micro":
            target = target.view(-1)
            input = input.view(-1)

        return self.update(input, target, weight, *args, **kwargs)

    def compute(self) -> torch.Tensor:
        return self.compute()
