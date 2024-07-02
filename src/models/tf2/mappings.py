from tensorflow.keras import metrics

activation_names = [
    "elu", "exponential", "gelu", "hard_sigmoid", "hard_silu", "hard_swish", "leaky_relu", "linear",
    "log_softmax", "mish", "relu", "relu6", "selu", "sigmoid", "silu", "softmax", "softplus",
    "softsign", "swish", "tanh"
]

metric_mapping = {
    "roc_auc": lambda *args, **kwargs: metrics.AUC(*args, curve="ROC", **kwargs),
    "pr_auc": lambda *args, **kwargs: metrics.AUC(*args, curve="ROC", **kwargs),
    "accuracy": "accuracy",
    "acc": "acc",
    "binary_accuracy": "binary_accuracy",
    "categorical_accuracy": "categorical_accuracy",
    "sparse_categorical_accuracy": "sparse_categorical_accuracy",
    "AUC": "AUC",
    "precision": "precision",
    "recall": "recall",
    "specificity_at_sensitivity": "specificity_at_sensitivity",
    "sensitivity_at_specificity": "sensitivity_at_specificity",
    "hinge": "hinge",
    "squared_hinge": "squared_hinge",
    "top_k_categorical_accuracy": "top_k_categorical_accuracy",
    "sparse_top_k_categorical_accuracy": "sparse_top_k_categorical_accuracy",
    "mean_absolute_error": "mean_absolute_error",
    "mae": "mae",
    "mean_squared_error": "mean_squared_error",
    "mse": "mse",
    "mean_absolute_percentage_error": "mean_absolute_percentage_error",
    "mape": "mape",
    "mean_squared_logarithmic_error": "mean_squared_logarithmic_error",
    "msle": "msle",
    "huber": "huber",
    "logcosh": "logcosh",
    "binary_crossentropy": "binary_crossentropy",
    "categorical_crossentropy": "categorical_crossentropy",
    "sparse_categorical_crossentropy": "sparse_categorical_crossentropy",
    "cosine_proximity": "cosine_proximity",
    "cosine_similarity": "cosine_similarity"
}
