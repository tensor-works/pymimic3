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

    def __init__(self, bining: Literal["log", "custom"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._binning = bining
        if self._binning == "custom":
            self._means = torch.tensor(CustomBins.means, dtype=torch.float32)
        elif self._binning == "log":
            self._means = torch.tensor(LogBins.means, dtype=torch.float32)
        else:
            raise ValueError(f"Binning must be one of 'log' or 'custom' but is {bining}.")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        prediction_means = self._means[torch.argmax(preds, axis=1)]
        if target.dim() > 1:
            target = torch.argmax(target, axis=1)
        target_means = self._means[target]
        super().update(prediction_means, target_means)

    def to(self, *args, **kwargs):
        self._means = self._means.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class BinaryAUPRC(_BinaryAUPRC):

    def update(self, predictions, labels):
        # Reshape predictions and labels to handle the batch dimension
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        super().update(predictions, labels)


class MulticlassAUPRC(_MulticlassAUPRC):

    def update(self, predictions, labels):
        # Reshape predictions and labels to handle the batch dimension
        labels = labels.view(-1)
        super().update(predictions, labels)


class AUPRC(metric.Metric[torch.Tensor]):

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


if __name__ == "__main__":
    # Multi-class classification data
    import torch
    # Compute precision-recall curve
    import numpy as np
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

    # -- Testing the MAE --
    y_true_mc = torch.torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
                                    [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]]).int()
    y_pred_mc = torch.torch.Tensor([0, 1, 0, 0, 2, 2, 0, 2, 0, 2]).int()

    mae_log = BinedMAE("log")
    mae_log.update(y_pred_mc, y_true_mc)
    mae_custom = BinedMAE("custom")
    mae_custom.update(y_pred_mc, y_true_mc)
    print("Log MAE:", mae_log.compute())
    print("Custom MAE:", mae_custom.compute())

    #
    y_true_multi = torch.torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
                                       [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0,
                                                                                    1]]).int()
    y_pred_multi = torch.torch.Tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6],
                                       [0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.1, 0.7],
                                       [0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.1, 0.7],
                                       [0.7, 0.2, 0.1]])

    # ---------------- Comparing Micro-Macro PR AUC using torch with sklearn --------------------
    print("--- Comparing Micro-Macro ROC AUC ---")

    # Compute ours: micro
    micro_rocauc = AUROC(task="multilabel", average="micro", num_labels=3)
    micro_rocauc.update(y_pred_multi, y_true_multi)
    print("Micro AUCROC (torch):", micro_rocauc.compute())

    # Compute ours: macro
    macro_rocauc = AUROC(task="multilabel", average="macro", num_labels=3)
    macro_rocauc.update(y_pred_multi, y_true_multi)
    print("Micro AUCPRC (torch):", macro_rocauc.compute())

    # Compute theirs
    # Flatten y_true_multi as numpy
    y_true_multi_flat = y_true_multi.numpy().flatten()
    y_pred_multi_flat = y_pred_multi.numpy().flatten()

    # Compute micro-average ROC AUC using sklearn
    micro_rocauc_sklearn = roc_auc_score(y_true_multi,
                                         y_pred_multi,
                                         average='micro',
                                         multi_class='ovr')
    print(f'Micro-average auc-roc (sklearn): {micro_rocauc_sklearn:.4f}')

    # Compute macro-average ROC AUC using sklearn
    macro_rocauc_sklearn = roc_auc_score(y_true_multi,
                                         y_pred_multi,
                                         average='macro',
                                         multi_class='ovr')
    print(f'Macro-average auc-roc (sklearn): {macro_rocauc_sklearn:.4f}')

    # ---------------- Comparing Micro-Macro PR AUC using torch with sklearn --------------------
    print("--- Comparing Micro-Macro PR AUC ---")
    micro_prauc = AUPRC(task="multilabel", num_labels=3, average="micro")
    macro_prauc = AUPRC(task="multilabel", num_labels=3, average="macro")

    # Compute ours
    for idx in range(len(y_true_multi)):
        yt = y_true_multi[idx, :].unsqueeze(0)
        yp = y_pred_multi[idx, :].unsqueeze(0)
        micro_prauc.update(yp, yt)
        macro_prauc.update(yp, yt)

    print("Micro AUCPR (torch):", micro_prauc.compute())
    print("Macro AUCPR (torch):", macro_prauc.compute())

    # Compute theirs
    roc_pr_list = []
    roc_auc_list = []

    # Iterate over each class
    for i in range(y_true_multi.shape[1]):
        y_true = y_true_multi[:, i]
        y_pred = y_pred_multi[:, i]

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        roc_pr_list.append(auc(recall, precision))

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_true, y_pred)
        roc_auc_list.append(roc_auc)

    print(f"PR AUC macro Score (sklearn): {np.mean(roc_pr_list)}")

    precision, recall, _ = precision_recall_curve(y_true_multi_flat, y_pred_multi_flat)
    pr_auc = auc(recall, precision)

    # Print results
    print(f"PR AUC micro Score (sklearn): {pr_auc}")
    print()
    # ---------------- Comparing Binary PR AUC using torch with sklearn --------------------
    prauc = AUPRC(task="binary")
    prauc.update(y_pred_multi.flatten(), y_true_multi.flatten())

    print("Binary AUCPR (torch):", prauc.compute())
    from torcheval.metrics.functional import binary_auprc
    binary_auprc(y_pred_multi.flatten(), y_true_multi.flatten())
    print("Binary AUCPR functional (torch):", prauc.compute())
    precision, recall, _ = precision_recall_curve(y_true_multi_flat, y_pred_multi_flat)
    pr_auc = auc(recall, precision)

    # Print results
    print(f"Binary PRAUC Score (sklearn): {pr_auc}")
