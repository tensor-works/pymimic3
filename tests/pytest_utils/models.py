import numpy as np
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
                        mask: np.ndarray = None):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X
    y_pred = model.predict(inputs)
    if task_name == 'LOS':
        assert_valid_metric_LOS(X=X, y_true=y_true, y_pred=y_pred, model=model, mask=mask)
    elif task_name in ['IHM', 'DECOMP']:
        assert_valid_metric_binary(X=X, y_true=y_true, y_pred=y_pred, model=model, mask=mask)
    elif task_name == 'PHENO':
        assert_valid_metric_PHENO(X=X, y_true=y_true, y_pred=y_pred, model=model, mask=mask)
    else:
        raise ValueError(f"Unknown task name: {task_name}")


def assert_valid_metric_LOS(X: np.ndarray,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            model: AbstractTf2Model,
                            mask: np.ndarray = None):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X
    loss, kappa, mae = model.evaluate(inputs, y_true, batch_size=len(X))
    print(f'Kappa: {kappa}\nMAE: {mae}')

    y_pred_classes = np.argmax(y_pred, axis=1)
    sklearn_kappa = cohen_kappa_score(y_true, y_pred_classes, labels=range(10))
    y_true_binned = np.array(CustomBins.means)[y_true]
    y_pred_binned = np.array(CustomBins.means)[y_pred_classes]
    sklearn_mae = mean_absolute_error(y_true_binned, y_pred_binned)

    print(f'Diff Kappa (sklearn-tf2): {sklearn_kappa - kappa}\n'
          f'Diff MAE (sklearn-tf2): {sklearn_mae - mae}')

    assert np.isclose(sklearn_mae, mae), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn MAE: {sklearn_mae}, Custom MAE: {mae}")
    assert np.isclose(sklearn_kappa, kappa), \
        (f"Sklearn Kappa: {sklearn_kappa}, Custom Kappa: {kappa}")
    print(f'Succeeded in comparing tf2 to sklearn metrics!')


def assert_valid_metric_binary(X: np.ndarray,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               model: AbstractTf2Model,
                               mask: np.ndarray = None):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X
    loss, roc_auc, roc_pr = model.evaluate(inputs, y_true, batch_size=len(X))
    y_pred = np.round(y_pred)
    print(f'ROC AUC: {roc_auc}\nROC PR: {roc_pr}')

    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    sklearn_rocauc = roc_auc_score(y_true, y_pred)
    print(f'ROC AUC (sklearn): {sklearn_rocauc:.4f}')
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    sklearn_pr_auc = auc(recall, precision)
    print(f'PR AUC (sklearn): {sklearn_pr_auc:.4f}')

    print(f'Diff ROC (sklearn-tf2): {sklearn_rocauc - roc_auc}\n'
          f'Diff PR (sklearn-tf2): {sklearn_pr_auc - roc_pr}')

    # High deviation when tf2 approximating AUC close to 1
    # https://stackoverflow.com/questions/52228899/keras-auc-on-validation-set-during-training-does-not-match-with-sklearn-auc
    assert np.isclose(sklearn_rocauc, roc_auc, atol=0.09), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn ROC-AUC: {sklearn_rocauc}, tf2 ROC-AUC: {roc_auc}")
    assert np.isclose(sklearn_pr_auc, roc_pr, atol=0.09), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn PR-AUC: {sklearn_pr_auc}, tf2 PR-AUC: {roc_pr}")
    print(f'Succeeded in comparing tf2 to sklearn metrics!')


def assert_valid_metric_PHENO(X: np.ndarray,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model: AbstractTf2Model,
                              mask: np.ndarray = None):
    if mask is not None:
        inputs = [X, mask]
    else:
        inputs = X
    loss, micro_roc_auc, macro_roc_auc = model.evaluate(inputs, y_true, batch_size=len(X))
    print(f'Micro ROC AUC: {micro_roc_auc}\nMacro ROC AUC: {macro_roc_auc}')

    micro_rocauc_sklearn = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')
    print(f'Micro-average auc-roc (sklearn): {micro_rocauc_sklearn:.4f}')
    macro_rocauc_sklearn = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    print(f'Macro-average auc-roc (sklearn): {macro_rocauc_sklearn:.4f}')

    print(f'Diff Micro ROC AUC Score: {micro_rocauc_sklearn - micro_roc_auc}\n'
          f'Diff Macro ROC AUC Score: {macro_rocauc_sklearn - macro_roc_auc}')

    assert np.isclose(micro_rocauc_sklearn, micro_roc_auc, atol=0.05), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn micro ROC-AUC: {micro_rocauc_sklearn}, tf2 micro ROC-AUC: {micro_roc_auc}")
    assert np.isclose(macro_rocauc_sklearn, macro_roc_auc, atol=0.05), \
        (f"Diverging results for sklearn and tensorflow metric. Sklearn macro ROC-AUC: {micro_rocauc_sklearn}, tf2 macro ROC-AUC: {micro_roc_auc}")
