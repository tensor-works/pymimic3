import tensorflow as tf
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, cohen_kappa_score
from metrics.tf2 import BinedMAE, CohenKappa, AUC
from metrics import CustomBins, LogBins
from utils.IO import *

# Multiclass classification labels (one-hot encoded)
y_true_mc = tf.constant(
    [
        [1, 0, 0],  # Class 0
        [0, 1, 0],  # Class 1
        [0, 0, 1],  # Class 2
        [1, 0, 0],  # Class 0
        [0, 1, 0],  # Class 1
        [0, 0, 1],  # Class 2
        [1, 0, 0],  # Class 0
        [0, 0, 1],  # Class 2
        [1, 0, 0],  # Class 0
        [0, 0, 1],  # Class 2
    ],
    dtype=tf.int64)

# Predicted class indices for multiclass classification
y_pred_mc = tf.constant(
    [
        [0.9, 0.0, 0.1],  # Class 0 
        [0.3, 0.4, 0.3],  # Class 1 
        [0.5, 0.4, 0.1],  # Class 0 
        [0.8, 0.0, 0.2],  # Class 0 
        [0.1, 0.0, 0.9],  # Class 2
        [0.3, 0.0, 0.7],  # Class 2 
        [0.5, 0.4, 0.1],  # Class 0 
        [0.4, 0.1, 0.5],  # Class 2 
        [0.6, 0.1, 0.3],  # Class 0 
        [0.2, 0.1, 0.7]  # Class 2
    ],
    dtype=tf.float32)

y_pred_mc_ordinal = tf.constant([0, 1, 0, 0, 2, 2, 0, 2, 0, 2], dtype=tf.int64)

# Convert predicted class indices to one-hot encoding
y_pred_mc_one_hot = tf.one_hot(y_pred_mc_ordinal, depth=3, dtype=tf.int64)

# Multilabel classification labels (one-hot encoded)
y_true_ml = tf.constant(
    [
        [1, 0, 0],  # Only class 0
        [0, 1, 0],  # Only class 1
        [0, 0, 1],  # Only class 2
        [1, 0, 0],  # Only class 0
        [0, 1, 0],  # Only class 1
        [0, 0, 1],  # Only class 2
        [1, 0, 0],  # Only class 0
        [0, 0, 1],  # Only class 2
        [1, 0, 0],  # Only class 0
        [1, 0, 1],  # Classes 0 and 2
    ],
    dtype=tf.int64)

# Predicted probabilities for multilabel classification
y_pred_ml = tf.constant(
    [
        [0.8, 0.1, 0.1],  # Only class 0
        [0.2, 0.7, 0.1],  # Only class 1
        [0.1, 0.3, 0.6],  # Only class 2
        [0.7, 0.2, 0.1],  # Only class 0 
        [0.1, 0.6, 0.3],  # Only class 1 
        [0.2, 0.1, 0.7],  # Only class 2
        [0.6, 0.3, 0.1],  # Only class 0
        [0.3, 0.5, 0.2],  # Only class 1
        [0.2, 0.1, 0.7],  # Only class 2
        [0.7, 0.2, 0.1],  # Only class 0
    ],
    dtype=tf.float32)

# Flattened versions for sklearn calculations
y_true_flat_ml = y_true_ml.numpy().flatten()
y_pred_flat_ml = y_pred_ml.numpy().flatten()


@pytest.mark.parametrize("bining_falvour", ["log", "custom"])
def test_bined_mae(bining_flavour):
    tests_io(f"Test Case: Bined MAE for binning flavour '{bining_flavour}'", level=0)

    # Compute custom tf2 MAE
    mae_metric = BinedMAE(binning=bining_flavour)
    mae_metric.update_state(y_true_mc, y_pred_mc_one_hot)

    # Compute gt MAE
    mae_result = mae_metric.result().numpy()

    # Ground truth calculation
    means = CustomBins.means if bining_flavour == "custom" else LogBins.means
    means = np.array(means)

    # Convert one-hot labels to class indices
    y_true_indices = tf.argmax(y_true_mc, axis=1).numpy()
    y_pred_indices = tf.argmax(y_pred_mc_one_hot, axis=1).numpy()

    # Compute ground truth MAE
    errors = np.abs(means[y_true_indices] - means[y_pred_indices])
    mae_gt = np.mean(errors)

    # Assert that the metric result matches the ground truth
    assert np.isclose(mae_result, mae_gt), (f"Metric result and ground truth do not match:\n"
                                            f"Bined MAE ({bining_flavour}): {mae_result}\n"
                                            f"Ground truth MAE: {mae_gt}\n"
                                            f"Difference: {mae_result - mae_gt}")
    tests_io("Successfully compared to ground truth. Test case passed.\n")


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_roc_auc_multilabel(average):
    tests_io(f"Test Case: Multilabel ROC AUC with '{average}' average", level=0)

    # Compute custom tf2 ROC AUC
    if average == "micro":
        roc_auc_metric = AUC(curve='ROC', average='micro')
    elif average == "macro":
        roc_auc_metric = AUC(curve='ROC', average='macro', multi_label=True, num_labels=3)

    roc_auc_metric.update_state(y_true_ml, y_pred_ml)
    roc_auc_result = roc_auc_metric.result().numpy()

    # Compute ROC AUC using scikit-learn
    y_true_ml_np = y_true_ml.numpy()
    y_pred_ml_np = y_pred_ml.numpy()
    roc_auc_sklearn = roc_auc_score(y_true_ml_np, y_pred_ml_np, average=average)

    assert np.isclose(roc_auc_sklearn, roc_auc_result), (\
        f"{average.capitalize()}-average auc-roc (tf2): {roc_auc_result}\n"
        f"{average.capitalize()}-average auc-roc (sklearn): {roc_auc_sklearn:.4f}\n",
        f"Diff: {roc_auc_result - roc_auc_sklearn}")
    tests_io("Successfully compared ROC AUC to sklearn ground truth. Test case passed.\n")


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_pr_auc_multilabel(average):
    tests_io(f"Test Case: Multilabel PR AUC with '{average}' average", level=0)

    # Compute custom tf2 ROC AUC
    if average == "micro":
        pr_auc_metric = AUC(curve='PR', average='micro')
    elif average == "macro":
        pr_auc_metric = AUC(curve='PR', average='macro', multi_label=True, num_labels=3)

    pr_auc_metric.update_state(y_true_ml, y_pred_ml)
    pr_auc_result = pr_auc_metric.result().numpy()

    # Compute sklearn PR AUC
    if average == "macro":
        prauc_per_class = []

        for i in range(y_true_ml.shape[1]):
            y_true = y_true_ml[:, i]
            y_pred = y_pred_ml[:, i]

            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            prauc_per_class.append(auc(recall, precision))
        pr_auc_sklearn = np.mean(prauc_per_class)

    else:
        precision, recall, _ = precision_recall_curve(y_true_flat_ml, y_pred_flat_ml)
        pr_auc_sklearn = auc(recall, precision)

    # Allow a small tolerance due to numerical differences
    # TODO! computation of PR AUC from sklearn seems to diverge from that of tf2
    # TODO! https://stackoverflow.com/questions/77037814/explaining-roc-auc-pr-auc-disparity-between-sklearn-and-tensorflow
    assert np.isclose(pr_auc_result, pr_auc_sklearn,
                      atol=0.04), (f"Metric result and sklearn result do not match:\n"
                                   f"PR AUC ({average}): {pr_auc_result}\n"
                                   f"Sklearn PR AUC ({average}): {pr_auc_sklearn}\n"
                                   f"Difference: {pr_auc_result - pr_auc_sklearn}")
    tests_io("Successfully compared PR AUC to sklearn ground truth. Test case passed.\n")


# Testing Cohen Kappa metric
def test_cohen_kappa():
    # TODO! test with sparse labels
    tests_io("Test Case: Cohen Kappa", level=0)

    # Compute tf2 PR AUC
    num_classes = 3
    kappa_metric = CohenKappa(num_classes=num_classes)

    # Convert one-hot labels to class indices
    y_true_ordinal = tf.argmax(y_true_mc, axis=1)
    y_pred_ordinal = tf.argmax(y_pred_mc_one_hot, axis=1)

    # Update the metric with the test data
    kappa_metric.update_state(y_true_ordinal, y_pred_mc)

    # Compute the metric result
    kappa_result = kappa_metric.result().numpy()

    # Compute Cohen Kappa using scikit-learn
    y_true_np = y_true_ordinal.numpy()
    y_pred_np = y_pred_ordinal.numpy()
    kappa_sklearn = cohen_kappa_score(y_true_np, y_pred_np)

    # Assert that the metric result matches the sklearn result
    assert np.isclose(kappa_result, kappa_sklearn), ( \
        f"Metric result and sklearn result do not match:\n"
        f"Cohen Kappa (tf2): {kappa_result}\n"
        f"Cohen Kappa (sklearn): {kappa_sklearn}\n"
        f"Difference: {kappa_result - kappa_sklearn}")
    tests_io("Successfully compared Cohen Kappa to sklearn ground truth. Test case passed.\n")


if __name__ == "__main__":
    # Test BinedMAE with both 'log' and 'custom' binning flavours
    for binning in ["log", "custom"]:
        test_bined_mae(binning)

    # Test ROC AUC and PR AUC with both 'micro' and 'macro' averages
    for average in ["micro", "macro"]:
        test_roc_auc_multilabel(average)
        test_pr_auc_multilabel(average)

    # Test Cohen Kappa metric
    test_cohen_kappa()
