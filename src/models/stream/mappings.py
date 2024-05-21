import river.metrics as metrics
from metrics.stream import PRAUC, MacroROCAUC, MicroROCAUC, LOSCohenKappa, LOSClassificationReport

metric_mapping = {
    "accuracy": metrics.Accuracy,
    "precision": metrics.Precision,
    "recall": metrics.Recall,
    "f1": metrics.F1,
    "roc_auc": metrics.ROCAUC(n_thresholds=20),
    "log_loss": metrics.LogLoss,
    "mae": metrics.MAE,
    "mse": metrics.MSE,
    "rmse": metrics.RMSE,
    "r2": metrics.R2,
    "mape": metrics.MAPE,
    "smape": metrics.SMAPE,
    "confusion_matrix": metrics.ConfusionMatrix,
    "classification_report": metrics.ClassificationReport,
    "los_classification_report": LOSClassificationReport,
    "cohen_kappa": LOSCohenKappa(weights="linear"),
    "pr_auc": PRAUC(n_thresholds=20),
    "micro_roc_auc": MicroROCAUC(n_thresholds=20),
    "macro_roc_auc": MacroROCAUC(n_thresholds=20)
}
