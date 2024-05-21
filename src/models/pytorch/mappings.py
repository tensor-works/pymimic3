import torch.optim as optim
import torch.nn as nn
import torchmetrics

__all__ = ["optimizer_mapping", "loss_mapping", "metric_mapping", "activation_mapping"]

optimizer_mapping = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adamax": optim.Adamax,
    "rmsprop": optim.RMSprop,
    "nadam": optim.NAdam
}

loss_mapping = {
    "binary_crossentropy": nn.BCELoss,
    "logits_binary_crossentropy": nn.BCEWithLogitsLoss,
    "categorical_crossentropy": nn.CrossEntropyLoss,
    "kld": nn.KLDivLoss,
    "mean_squared_error": nn.MSELoss,
    "mean_absolute_error": nn.L1Loss,
    "hinge": nn.HingeEmbeddingLoss,
    "poisson": nn.PoissonNLLLoss,
    "cosine_similarity": nn.CosineEmbeddingLoss,
    "huber": nn.SmoothL1Loss,
}

metric_mapping = {
    "accuracy": torchmetrics.Accuracy,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
    "f1": torchmetrics.F1Score,
    "roc_auc": torchmetrics.AUROC,
    "msl_error": torchmetrics.MeanSquaredLogError,
    "mean_squared_error": torchmetrics.MeanSquaredError,
    "mean_absolute_error": torchmetrics.MeanAbsoluteError,
    "mae": torchmetrics.MeanAbsoluteError,
    "mse": torchmetrics.MeanSquaredError,
    "r2": torchmetrics.R2Score,
    "mape": torchmetrics.MeanAbsolutePercentageError,
    "confusion_matrix": torchmetrics.ConfusionMatrix,
    "cohen_kappa": torchmetrics.CohenKappa,
    "pr_auc": torchmetrics.PrecisionRecallCurve
}

activation_mapping = {"sigmoid": nn.Sigmoid(), "softmax": nn.Softmax(dim=-1), "relu": nn.ReLU()}
