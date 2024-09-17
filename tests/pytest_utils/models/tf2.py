import numpy as np
from typing import Dict
from generators.tf2 import TFGenerator
from utils.arrays import zeropad_samples


def unroll_generator(generator: TFGenerator, deep_supervision: bool = False):
    if deep_supervision:
        X, y = list(zip(*[(X, y) for X, y in iter(generator)]))
        X, M = list(zip(*X))
        X = zeropad_samples(X, axis=1)
        M = zeropad_samples(M, axis=1)
        y_true = zeropad_samples(y, axis=1)
        return X, M, y_true
    X, y = list(zip(*[(X, y) for X, y in iter(generator)]))
    X = zeropad_samples(X, axis=1)
    y_true = np.concatenate(y)
    return X, y_true


def assert_model_performance(history, task: str, target_metrics: Dict[str, float]):

    for metric, target_value in target_metrics.items():
        if metric in ["loss", "custom_mae"]:
            actual_value = min(history.history[metric])
            comparison = actual_value <= target_value
        else:
            # For other metrics, assume higher is better unless it's an error metric
            actual_value = max(history.history[metric])
            comparison = actual_value >= target_value if "error" not in metric.lower(
            ) and "loss" not in metric.lower() else actual_value <= target_value

        assert comparison, \
            (f"Failed in asserting {metric} ({actual_value:.4f}) "
             f"{'<=' if 'loss' in metric.lower() or 'error' in metric.lower() else '>='} {target_value} for task {task}")
