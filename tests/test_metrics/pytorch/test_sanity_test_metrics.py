import torch
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from metrics.pytorch import BinedMAE, AUPRC, AUROC
from metrics import CustomBins, LogBins
from utils.IO import *

# The error is 0-2, 1-2 at 2, 4
y_true_mc = torch.torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]]).int()
y_pred_mc = torch.torch.Tensor([0, 1, 0, 0, 2, 2, 0, 2, 0, 2]).int()

# The error is
y_true_ml = torch.torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]).int()
y_pred_ml = torch.torch.Tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.7, 0.2, 0.1],
                                [0.1, 0.6, 0.3], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1], [0.3, 0.5, 0.2],
                                [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])

y_true_flat_mc = y_true_ml.numpy().flatten()
y_pred_flat_mc = y_pred_ml.numpy().flatten()


@pytest.mark.parametrize("bining_falvour", ["log", "custom"])
def test_bined_mae(bining_falvour):
    tests_io(f"Test Case: Bined MAE for bining flavour {bining_falvour}")

    # Compute custom torch MAE
    mae_from_one_hot = BinedMAE(bining_falvour)
    mae_from_one_hot.update(y_pred_mc, y_true_mc)

    # Compute gt MAE
    means = CustomBins.means if bining_falvour == "custom" else LogBins.means
    means = np.array(means)
    mae_gt = np.abs(means[[0, 1]] - means[[2, 2]]).sum() / 10

    assert np.isclose(mae_gt, mae_from_one_hot.compute()), (
        f"Bined MAE ({bining_falvour}): {mae_from_one_hot.compute()}\n"
        f"Ground truth: {mae_gt}\n"
        f"Diff: {mae_from_one_hot.compute() - mae_gt}")
    tests_io("Successfully compared to ground truth. Test case passed.")


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_roc_auc_multilabel(average):
    tests_io(f"Test Case: multilabel ROC AUC with {average} average")

    # Compute custom torch ROC AUC
    rocauc = AUROC(task="multilabel", average=average, num_labels=3)
    rocauc.update(y_pred_ml, y_true_ml)

    # Compute ROC AUC using sklearn
    rocauc_sklearn = roc_auc_score(y_true_ml, y_pred_ml, average=average, multi_class='ovr')

    assert np.isclose(rocauc_sklearn, rocauc.compute()), (\
        f"{average.capitalize()}-average auc-roc (torch): {rocauc.compute()}\n"
        f"{average.capitalize()}-average auc-roc (sklearn): {rocauc_sklearn:.4f}\n",
        f"Diff: {rocauc.compute() - rocauc_sklearn}")
    tests_io("Successfully compared ROC AUC to sklearn ground truth. Test case passed.")


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_pr_auc_multilabel(average):
    tests_io(f"Test Case: multilabel PR AUC with {average} average")

    prauc = AUPRC(task="multilabel", num_labels=3, average=average)

    # Compute torch PR AUC
    for idx in range(len(y_true_ml)):
        yt = y_true_ml[idx, :].unsqueeze(0)
        yp = y_pred_ml[idx, :].unsqueeze(0)
        prauc.update(yp, yt)

    # Compute sklearn PR AUC
    if average == "macro":
        prauc_per_class = []

        for i in range(y_true_ml.shape[1]):
            y_true = y_true_ml[:, i]
            y_pred = y_pred_ml[:, i]

            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            prauc_per_class.append(auc(recall, precision))
        prauc_sklearn = np.mean(prauc_per_class)

    else:
        precision, recall, _ = precision_recall_curve(y_true_flat_mc, y_pred_flat_mc)
        prauc_sklearn = auc(recall, precision)

    np.isclose(prauc.compute(), prauc_sklearn), (\
        f"{average.capitalize()}-averge PR AUC (torch): {prauc.compute()}\n"
        f"{average.capitalize()}-averge PR AUC (sklearn): {prauc_sklearn}\n"
        f"Diff: {prauc.compute() - prauc_sklearn}")

    tests_io("Successfully compared PR AUC to sklearn ground truth. Test case passed.")


if __name__ == "__main__":
    for bining in ["log", "custom"]:
        test_bined_mae(bining)

    for average in ["micro", "macro"]:
        test_roc_auc_multilabel(average)
        test_pr_auc_multilabel(average)
