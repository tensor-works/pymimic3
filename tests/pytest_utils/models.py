import numpy as np
import torch
from metrics import CustomBins
from utils.IO import *
from tests.tsettings import *
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, roc_auc_score, precision_recall_curve, auc
from models.tf2 import AbstractTf2Model
from tests.msettings import *


def assert_valid_metric(X: np.ndarray,
                        y_true: np.ndarray,
                        task_name: str,
                        model: AbstractTf2Model,
                        mask: np.ndarray = None,
                        flavour: str = 'tf2'):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X
    if hasattr(model, 'predict'):
        y_pred = model.predict(inputs)
    else:
        # TODO add predict to our trainer
        y_pred = model(torch.tensor(inputs)).to("cpu").detach().numpy()
    if task_name == 'LOS':
        assert_valid_metric_LOS(X=X, y_true=y_true, y_pred=y_pred, model=model, mask=mask)
    elif task_name in ['IHM', 'DECOMP']:
        assert_valid_metric_binary(X=X, y_true=y_true, y_pred=y_pred, model=model, mask=mask)
    elif task_name == 'PHENO':
        assert_valid_metric_PHENO(X=X, y_true=y_true, y_pred=y_pred, model=model, mask=mask)
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    print(f'Succeeded in comparing {flavour} to sklearn metrics!')


def assert_valid_metric_LOS(X: np.ndarray,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            model: AbstractTf2Model,
                            mask: np.ndarray = None):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X

    # Evaluate tf2
    loss, kappa, mae = model.evaluate(inputs, y_true, batch_size=len(X))

    y_pred = np.argmax(y_pred, axis=-1)

    # Squeeze it
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # Apply mask
    if mask is not None:
        mask = np.squeeze(mask)
        y_true = y_true[mask]
        if y_pred.shape[-1] != mask.shape[-1]:
            mask = np.repeat(mask, y_pred.shape[-1], axis=-1)
        y_pred = y_pred[mask]

    # Evaluate sklearn
    sklearn_kappa = cohen_kappa_score(y_true, y_pred, labels=range(10))
    y_true_binned = np.array(CustomBins.means)[y_true]
    y_pred_binned = np.array(CustomBins.means)[y_pred]
    sklearn_mae = mean_absolute_error(y_true_binned, y_pred_binned)
    info_io(f'Kappa: {kappa}\nMAE: {mae}\n'
            f'Cohen Kappa (sklearn): {sklearn_kappa:.4f}\n'
            f'Binned MAE (sklearn): {sklearn_mae:.4f}\n'
            f'Diff Kappa (sklearn-tf2): {sklearn_kappa - kappa}\n'
            f'Diff MAE (sklearn-tf2): {sklearn_mae - mae}')

    # Assert closeness
    assert np.isclose(sklearn_mae, mae, atol=0.09), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn MAE: {sklearn_mae}, Custom MAE: {mae}")
    assert np.isclose(sklearn_kappa, kappa, atol=0.09), \
        (f"Sklearn Kappa: {sklearn_kappa}, Custom Kappa: {kappa}")


def assert_valid_metric_binary(X: np.ndarray,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               model: AbstractTf2Model,
                               mask: np.ndarray = None):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X

    # Evaluate tf2
    loss, roc_auc, roc_pr = model.evaluate(inputs, y_true, batch_size=len(X))

    # Apply mask
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # Squeeze it
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # Evaluate sklearn
    sklearn_rocauc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    sklearn_pr_auc = auc(recall, precision)

    info_io(f'ROC AUC: {roc_auc}\nROC PR: {roc_pr}\n'
            f'ROC AUC (sklearn): {sklearn_rocauc:.4f}\n'
            f'PR AUC (sklearn): {sklearn_pr_auc:.4f}\n'
            f'Diff ROC (sklearn-tf2): {sklearn_rocauc - roc_auc}\n'
            f'Diff PR (sklearn-tf2): {sklearn_pr_auc - roc_pr}')

    # Assert closeness
    # High deviation when tf2 approximating AUC close to 1
    # https://stackoverflow.com/questions/52228899/keras-auc-on-validation-set-during-training-does-not-match-with-sklearn-auc
    assert np.isclose(sklearn_rocauc, roc_auc, atol=0.09), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn ROC-AUC: {sklearn_rocauc}, tf2 ROC-AUC: {roc_auc}")
    assert np.isclose(sklearn_pr_auc, roc_pr, atol=0.09), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn PR-AUC: {sklearn_pr_auc}, tf2 PR-AUC: {roc_pr}")


def assert_valid_metric_PHENO(X: np.ndarray,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model: AbstractTf2Model,
                              mask: np.ndarray = None):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X

    # Evaluate tf2
    loss, micro_roc_auc, macro_roc_auc = model.evaluate(inputs, y_true, batch_size=len(X))

    # Squeeze it
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # Evaluate sklearn
    micro_rocauc_sklearn = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')
    macro_rocauc_sklearn = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')

    info_io(f'Micro ROC AUC: {micro_roc_auc}\nMacro ROC AUC: {macro_roc_auc}\n'
            f'Micro-average auc-roc (sklearn): {micro_rocauc_sklearn:.4f}\n'
            f'Macro-average auc-roc (sklearn): {macro_rocauc_sklearn:.4f}\n'
            f'Diff Micro ROC AUC Score: {micro_rocauc_sklearn - micro_roc_auc}\n'
            f'Diff Macro ROC AUC Score: {macro_rocauc_sklearn - macro_roc_auc}')

    # Assert closeness
    assert np.isclose(micro_rocauc_sklearn, micro_roc_auc, atol=0.05), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn micro ROC-AUC: {micro_rocauc_sklearn}, tf2 micro ROC-AUC: {micro_roc_auc}")
    assert np.isclose(macro_rocauc_sklearn, macro_roc_auc, atol=0.05), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn macro ROC-AUC: {micro_rocauc_sklearn}, tf2 macro ROC-AUC: {micro_roc_auc}")
