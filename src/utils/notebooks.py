import ipywidgets as widgets

# The docker environment manipulates the python path to include our source directory
# Execute this from within the docker environ to make these import work
import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import math

# The docker environment manipulates the python path to include our source directory
# Execute this from within the docker environ to make these import work
import visualization

label_name = {
    "hhourly": 'energy(kWh/hh)',
    "hourly": 'energy(kWh/hh)_sum',
    "hdaily": 'energy(kWh/hh)_sum',
    "qdaily": 'energy(kWh/hh)_sum',
    "daily": 'energy_sum'
}


class Metric():
    """
    """

    def __init__(self, bin_averages):
        """"""
        self.bin_averages = bin_averages

    def cat_mse(self, y_true, y_pred):
        """
        """
        y_pred = math.argmax(y_pred, axis=1)
        y_true = math.argmax(y_true, axis=1)
        y_pred = np.array([self.bin_averages[val] for val in y_pred.numpy()])
        y_true = np.array([self.bin_averages[val] for val in y_true.numpy()])

        return keras.losses.mse(y_pred, y_true)


class History():

    def __init__(self):
        self.acc = {}
        self.loss = {}
        self.mse = {}

    def update(self, value, history):
        self.acc[f"{value}_acc"] = history.history["accuracy"]
        self.acc[f"{value}_val_acc"] = history.history["val_accuracy"]

        self.loss[f"{value}_loss"] = history.history["loss"]
        self.loss[f"{value}_val_loss"] = history.history["val_loss"]

        self.mse[f"{value}_cat_mse"] = history.history["cat_mse"]
        self.mse[f"{value}_val_cat_mse"] = history.history["val_cat_mse"]


def make_hist_plot(values, history, type):
    plt.clf()
    colors = ["darkviolet", "darkgreen", "goldenrod", "darkblue"]
    plt.figure(figsize=(10, 5))

    for value, color in zip(values, colors):
        key = f"{value}_{type}"
        plt.plot(history[key], label=key, color=color)
        val_key = f"{value}_val_{type}"
        plt.plot(history[val_key], label=val_key, color=color, linestyle="--")

    plt.grid()
    plt.legend()