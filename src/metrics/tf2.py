import os
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Literal
from metrics import CustomBins, LogBins


class DynamicCohenKappa(tfa.metrics.CohenKappa):

    def __init__(self, **kwargs):
        # Initialize with a placeholder number of classes (e.g., 1)
        self._init_args = kwargs
        super().__init__(num_classes=2, **self._init_args)
        self._num_classes_set = False

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self._num_classes_set:
            # Determine the number of classes from y_true and y_pred
            num_classes = tf.shape(y_pred)[1]
            # Reinitialize the internal state with the correct number of classes
            super().__init__(num_classes=int(num_classes), **self._init_args)
            self._num_classes_set = True
        return super().update_state(y_true, y_pred, sample_weight)


class BinedMAE(tf.keras.metrics.MeanAbsoluteError):

    def __init__(self, binning: Literal["log", "custom"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._binning = binning
        if self._binning == "custom":
            self._means = tf.constant(CustomBins.means, dtype=tf.float32)
        elif self._binning == "log":
            self._means = tf.constant(LogBins.means, dtype=tf.float32)
        else:
            raise ValueError(f"Binning must be one of 'log' or 'custom' but is {binning}.")

    def update_state(self, y_true, y_pred, sample_weight=None):
        prediction_means = tf.gather(self._means, tf.argmax(y_pred, axis=1))

        if len(tf.shape(y_true)) > 1:
            y_true = tf.argmax(y_true, axis=1)
        target_means = tf.gather(self._means, y_true)

        return super().update_state(target_means, prediction_means, sample_weight)


class AUC(tf.keras.metrics.AUC):

    def __init__(self,
                 num_thresholds=50,
                 multi_label=False,
                 num_labels=None,
                 curve: Literal["PR", "ROC"] = 'ROC',
                 average: Literal["macro", "weighted", "none", "micro"] = "macro",
                 **kwargs):
        # When average is micro non-mulitlabel
        if num_labels and average != "micro":
            kwargs.update({'num_labels': num_labels})
            multi_label = True
        elif average == "micro":
            multi_label = False
            if "numl_labels" in kwargs:
                kwargs.pop("num_labels")
            if 'multi_label' in kwargs:
                kwargs.pop("multi_label")
        super().__init__(num_thresholds=num_thresholds,
                         curve=curve,
                         name=f'{average.lower()}_{curve.lower()}_auc',
                         multi_label=multi_label,
                         **kwargs)
        # Correct average
        if not average in ["macro", "weighted", "none", "micro"]:
            raise ValueError(
                f"Average must be one of macro, weighted, none or micro! Is {average}.")
        self._average = average
        self._multi_label = multi_label
        self._num_thresholds = num_thresholds
        self._curve = curve

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._average != "micro" and self._multi_label == False and len(
                y_pred.shape) > 1 and y_pred.shape[1] > 1:
            raise ValueError("Specified single lable prediction but got multi-label pred.")
        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_thresholds": self._num_thresholds,
            "curve": self._curve,
            "average": self._average
        })
        return config


if __name__ == "__main__":
    # Multi-class classification data
    import torch
    # Compute precision-recall curve
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import cohen_kappa_score

    # -- Testing the binned MAE --
    metric = BinedMAE(binning="custom")

    # Create some test data
    y_true = tf.constant([0, 2, 4, 1, 3])  # True labels (indices)
    y_pred = tf.constant([
        [0.9, 0.1, 0.0, 0.0, 0.0],  # Predicted as class 0
        [0.1, 0.1, 0.7, 0.1, 0.0],  # Predicted as class 2
        [0.0, 0.0, 0.1, 0.2, 0.7],  # Predicted as class 4
        [0.6, 0.3, 0.1, 0.0, 0.0],  # Predicted as class 0
        [0.1, 0.1, 0.1, 0.6, 0.1],  # Predicted as class 3
    ])

    # Update the metric
    metric.update_state(y_true, y_pred)

    # Compute the result
    result = metric.result().numpy()

    print(f"BinnedMAE result: {result}")

    # Reset the metric
    metric.reset_state()

    # Ground truth labels (one-hot encoded)
    y_true_multi = torch.Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
    ]).int()

    # Predicted probabilities (softmax applied)
    y_pred_multi = torch.Tensor([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.3, 0.6],
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.1, 0.7],
        [0.6, 0.3, 0.1],
        [0.3, 0.5, 0.2],
        [0.2, 0.1, 0.7],
        [0.7, 0.2, 0.1],
    ])

    dynamic = DynamicCohenKappa()
    dynamic.update_state(y_true_multi, y_pred_multi)
    print(f'Dynamic (tf2): {dynamic.result()}')

    y_true = y_true_multi.argmax(dim=1).numpy()
    y_pred = y_pred_multi.argmax(dim=1).numpy()
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa}")

    # Compute ours: micro
    micro_rocauc = AUC(curve='ROC', average="micro")
    micro_rocauc.update_state(y_true_multi, y_pred_multi)
    print(f'Micro-average (tf2): {micro_rocauc.result()}')

    # Compute ours: micro
    macro_rocauc = AUC(curve='ROC', average="macro", multi_label=True)
    macro_rocauc.update_state(y_true_multi, y_pred_multi)
    print(f'Macro-average (tf2): {macro_rocauc.result()}')

    metrics = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=3)
    metrics.update_state(y_true_multi, y_pred_multi)
    print(f'Base AUC: {metrics.result()}')

    # Compute theirs
    # Flatten y_true_multi as numpy
    y_true_multi_flat = y_true_multi.numpy().flatten()
    y_pred_multi_flat = y_pred_multi.numpy().flatten()

    metrics = tf.keras.metrics.AUC(curve='ROC')
    metrics.update_state(y_true_multi_flat, y_pred_multi_flat)
    print(f'Base micro: {metrics.result()}')

    metrics = tf.keras.metrics.AUC(curve='ROC')
    metrics.update_state(y_true_multi, y_pred_multi)
    print(f'Base AUC unflattened: {metrics.result()}')

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
