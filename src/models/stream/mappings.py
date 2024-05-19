import river.metrics as metrics

metric_mapping = {
    "accuracy": metrics.Accuracy,
    "precision": metrics.Precision,
    "recall": metrics.Recall,
    "f1": metrics.F1,
    "roc_auc": metrics.ROCAUC,
    "log_loss": metrics.LogLoss,
    "mae": metrics.MAE,
    "mse": metrics.MSE,
    "rmse": metrics.RMSE,
    "r2": metrics.R2,
    "mape": metrics.MAPE,
    "smape": metrics.SMAPE,
    "confusion_matrix": metrics.ConfusionMatrix
}
