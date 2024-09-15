from generators.pytorch import TorchGenerator
from utils.arrays import zeropad_samples

import numpy as np


def unroll_generator(generator: TorchGenerator, deep_supervision: bool = False):
    X, y = list(zip(*[(X, y) for X, y in iter(generator)]))
    if deep_supervision:
        X, M = list(zip(*X))
        M = [m.numpy() for m in M]
        M = zeropad_samples(M, axis=1)
    X = [x.numpy() for x in X]
    X = zeropad_samples(X, axis=1)
    if deep_supervision:
        y = [y_sample.numpy() for y_sample in y]
        y = zeropad_samples(y, axis=1)
        return X, M, y
    y = [y_sample for y_sample in y]
    y = np.concatenate(y)
    return X, y


def assert_model_performance(history, task, target_metrics):

    for metric, target_value in target_metrics.items():
        if "loss" in metric:
            actual_value = min(list(history[metric].values()))
            comparison = actual_value <= target_value
        elif "mae" in metric:
            actual_value = min(list(history["train_metrics"][metric].values()))
            comparison = actual_value <= target_value
        else:
            actual_value = max(list(history["train_metrics"][metric].values()))
            comparison = actual_value >= target_value

        assert comparison, \
            (f"Failed in asserting {metric} ({actual_value}) "
             f"{'<=' if 'loss' in metric or 'mae' in metric  else '>='} {target_value} for task {task}")
