import json
import pdb
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow import math
import tensorflow as tf
import numpy as np
from generators import AbstractBatchGenerator
from utils.IO import *
from functools import partial
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, confusion_matrix
from tensorflow import constant
from tensorflow.keras.metrics import AUC

import tensorflow as tf
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils import dict_subset

import numpy as np


class AbstractMIMICMetric(keras.callbacks.Callback):

    def __init__(self, train_generator: AbstractBatchGenerator,
                 val_generator: AbstractBatchGenerator, model: Model, metrics: list) -> None:
        """_summary_

        Args:
            train_generator (AbstractBatchGenerator): _description_
            test_generator (AbstractBatchGenerator): _description_
            model (Model): _description_
            metric (function): _description_
        """
        self._train_generator = train_generator
        self._val_generator = val_generator
        self._model = model
        self._possible_metrics = {
            "auc_roc": roc_auc_score,
            "auc_pr": lambda y_true, y_pred: auc(*precision_recall_curve(y_true, y_pred)[:2]),
            "auc_roc_micro": partial(roc_auc_score, average="micro"),
            "auc_roc_macro": partial(roc_auc_score, average="macro"),
            "accuracy": accuracy_score
        }
        self.__check_inputs(train_generator, val_generator, model, metrics)
        self._metrics = dict_subset(self._possible_metrics, metrics)

    def __check_inputs(self, train_generator, val_generator, model, metrics):
        """_summary_

        Args:
            train_generator (_type_): _description_
            val_generator (_type_): _description_
            model (_type_): _description_
            metrics (_type_): _description_
        """
        assert hasattr(train_generator, "next")
        assert hasattr(val_generator, "next")
        assert hasattr(model, "predict")
        assert all([metric in self._possible_metrics for metric in metrics])

    def on_epoch_end(self, batch, logs=None):
        """_summary_

        Args:
            batch (_type_): _description_
            logs (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        metric_message, error_message = self._compute_metric(self._train_generator)
        vale_metric_message, val_error_message = self._compute_metric(self._train_generator,
                                                                      val=True)
        tf_output_len = 54 + sum([len(log) + 12 for log in logs if log != "lr"])
        if batch == 1:
            print(metric_message)
        else:
            print(f"\033[{tf_output_len}C\033[1A - {metric_message} - {vale_metric_message}")
        for msg in error_message + val_error_message:
            info_io("Exception has occured in model evaluation")
            info_io(msg)
        return super().on_batch_end(batch, logs)

    def _compute_metric(self, generator: AbstractBatchGenerator, val: bool = False):
        """_summary_

        Args:
            generator (AbstractBatchGenerator): _description_
        """
        y_pred, y_true = zip(*[(lambda x, y: (np.round(self._model.predict(x)), y))(
            *generator.next()) for step in range(generator.steps)])
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        metric_message = list()
        error_message = list()
        for metric, func in self._metrics.items():
            try:
                metric_message.append(("val_" if val else "") +
                                      f"{metric}: {func(y_pred, y_true):.4f}")
            except ValueError as error:
                error_message.append(error)
        return " - ".join(metric_message), error_message


class DecompensationMetrics(keras.callbacks.Callback):

    def __init__(self,
                 train_data_gen,
                 val_data_gen,
                 deep_supervision,
                 batch_size=32,
                 early_stopping=True,
                 verbose=2):
        super(DecompensationMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.deep_supervision = deep_supervision
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (x, y) = next(data_gen)
            pred = self.model.predict(x, batch_size=self.batch_size)
            if self.deep_supervision:
                for m, t, p in zip(x[1].flatten(), y.flatten(), pred.flatten()):
                    if np.equal(m, 1):
                        y_true.append(t)
                        predictions.append(p)
            else:
                y_true += list(y.flatten())
                predictions += list(pred.flatten())
        print('\n')
        predictions = np.array(predictions)
        predictions = np.stack([1 - predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            max_auc = np.max([x["auroc"] for x in self.val_history])
            cur_auc = self.val_history[-1]["auroc"]
            if max_auc > 0.88 and cur_auc < 0.86:
                self.model.stop_training = True


class InHospitalMortalityMetrics(keras.callbacks.Callback):

    def __init__(self,
                 train_data,
                 val_data,
                 target_repl,
                 batch_size=32,
                 early_stopping=True,
                 verbose=2):
        super(InHospitalMortalityMetrics, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.target_repl = target_repl
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data, history, dataset, logs):
        y_true = []
        predictions = []
        B = self.batch_size
        for i in range(0, len(data[0]), B):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, len(data[0])), end='\r')
            if self.target_repl:
                (x, y, y_repl) = (data[0][i:i + B], data[1][0][i:i + B], data[1][1][i:i + B])
            else:
                (x, y) = (data[0][i:i + B], data[1][i:i + B])
            outputs = self.model.predict(x, batch_size=B)
            if self.target_repl:
                predictions += list(np.array(outputs[0]).flatten())
            else:
                predictions += list(np.array(outputs).flatten())
            y_true += list(np.array(y).flatten())
        print('\n')
        predictions = np.array(predictions)
        predictions = np.stack([1 - predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data, self.val_history, 'val', logs)

        if self.early_stopping:
            max_auc = np.max([x["auroc"] for x in self.val_history])
            cur_auc = self.val_history[-1]["auroc"]
            if max_auc > 0.85 and cur_auc < 0.83:
                self.model.stop_training = True


class PhenotypingMetrics(keras.callbacks.Callback):

    def __init__(self, train_data_gen, val_data_gen, batch_size=32, early_stopping=True, verbose=2):
        super(PhenotypingMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (x, y) = next(data_gen)
            outputs = self.model.predict(x, batch_size=self.batch_size)
            if data_gen.target_repl:
                y_true += list(y[0])
                predictions += list(outputs[0])
            else:
                y_true += list(y)
                predictions += list(outputs)
        print('\n')
        predictions = np.array(predictions)
        ret = metrics.print_metrics_multilabel(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            max_auc = np.max([x["ave_auc_macro"] for x in self.val_history])
            cur_auc = self.val_history[-1]["ave_auc_macro"]
            if max_auc > 0.75 and cur_auc < 0.73:
                self.model.stop_training = True


class LengthOfStayMetrics(keras.callbacks.Callback):

    def __init__(self,
                 train_data_gen,
                 val_data_gen,
                 partition,
                 batch_size=32,
                 early_stopping=True,
                 verbose=2):
        super(LengthOfStayMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.partition = partition
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (x, y_processed, y) = data_gen.next(return_y_true=True)
            pred = self.model.predict(x, batch_size=self.batch_size)
            if isinstance(x, list) and len(x) == 2:  # deep supervision
                if pred.shape[-1] == 1:  # regression
                    pred_flatten = pred.flatten()
                else:  # classification
                    pred_flatten = pred.reshape((-1, 10))
                for m, t, p in zip(x[1].flatten(), y.flatten(), pred_flatten):
                    if np.equal(m, 1):
                        y_true.append(t)
                        predictions.append(p)
            else:
                if pred.shape[-1] == 1:
                    y_true += list(y.flatten())
                    predictions += list(pred.flatten())
                else:
                    y_true += list(y)
                    predictions += list(pred)
        print('\n')
        if self.partition == 'log':
            predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
            ret = metrics.print_metrics_log_bins(y_true, predictions)
        if self.partition == 'custom':
            predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
            ret = metrics.print_metrics_custom_bins(y_true, predictions)
        if self.partition == 'none':
            ret = metrics.print_metrics_regression(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            max_kappa = np.max([x["kappa"] for x in self.val_history])
            cur_kappa = self.val_history[-1]["kappa"]
            max_train_kappa = np.max([x["kappa"] for x in self.train_history])
            if max_kappa > 0.38 and cur_kappa < 0.35 and max_train_kappa > 0.47:
                self.model.stop_training = True


class ConfusionMatrix(object):
    """
    """

    def __init__(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        return

    def score(self, X, y) -> float:
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._score_value = confusion_matrix(X, y)
        return self._score_value

    def partial_score(self, X, y) -> np.ndarray:
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._score_value += confusion_matrix(X, y)
        return self._score_value


class Meter():

    def __init__(self, storage_path=None):
        """
        """
        self.storage_path = storage_path

        self.print_switch = {
            "accuracy": self.print_accuracy,
            "auc_pr": partial(self.print_auc_roc, curve="PR"),
            "auc_roc": partial(self.print_auc_roc, curve="ROC"),
            "auc_roc_micro": partial(self.print_auc_roc, average="micro"),
            "auc_roc_macro": partial(self.print_auc_roc, average="macro"),
            "cohen_kappa": self.print_cohen_kappa,
            "mae": self.print_mean_absolut_error,
            "mse": self.print_mean_square_error,
            "confusion_matrix": self.print_confusion_matrix
        }

    def save(self):
        """
        """
        with open(Path(self.storage_path, f"history.json"), 'w') as save_file:
            json.dump(self.metrics, save_file)

    def print_metrics(self, prediction, y_true, metrics):
        """
        """
        returns = dict()
        for metric in metrics:
            current_metric = self.print_switch[metric](prediction, y_true)
            returns.update(current_metric)

        return returns

    def print_auc_roc(self, prediction, y_true, curve=None, average=None):
        """
        """
        if curve:
            auc = AUC(curve=curve)
            auc.update_state(prediction, y_true)
            info_io(f"AUC-{curve}: {auc.result().numpy()}")
            return {f"auc_{curve}": float(auc.result().numpy())}
        else:
            result = roc_auc_score(prediction, y_true, average="micro")
            info_io(f"AUC-{average}: {result}")
            return {f"auc_{average}": result}

    def print_accuracy(self, prediction, y_true):
        """
        """
        acc = accuracy_score(prediction, y_true)
        info_io(f"Accuracy: {acc}")
        return {"accuracy": acc}

    def print_cohen_kappa(self, prediction, y_true):
        """
        """
        kappa = cohen_kappa_score(prediction, y_true, weights='linear')
        info_io(f"Cohen Kappa: {kappa}")
        return {"kappa": kappa}

    def print_mean_absolut_error(self, prediction, y_true):
        """
        """
        mae = mean_absolute_error(prediction, y_true)
        info_io(f"Mean Absolute Error: {mae}")
        return {"mae": mae}

    def print_mean_square_error(self, prediction, y_true):
        """
        """
        mse = mean_squared_error(prediction, y_true)
        info_io(f"Mean Square Error: {mse}")
        return {"mse": mse}

    def print_confusion_matrix(self, prediction, y_true):
        """
        """
        conf_mat = confusion_matrix(prediction, y_true)
        info_io(f"Confusion Matrix: {conf_mat}\n")
        return {}


class MAPE(MeanAbsolutePercentageError):
    """
    """

    def __init__(self, name='custom', dtype=None):
        """
        """
        super().__init__(name, dtype)
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred = math.argmax(y_pred, axis=1)
        y_pred = tf.cast(y_pred, tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)


class CustomMSE(tf.keras.losses.Loss):

    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        return mse


class CustomCategoricalMSE():
    """
    """

    def __init__(self, bin_averages):
        """_summary_

        Args:
            bin_averages (_type_): _description_
        """
        self.bin_averages = bin_averages
        self.n_averages = len(bin_averages)

    def average_categorical_mse(self, y_true, y_pred):
        """_summary_

        Args:
            y_true (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = math.argmax(y_pred, axis=1)
        y_true = math.argmax(y_true, axis=1)

        try:
            y_pred = np.array([
                self.bin_averages[val]
                if val < self.n_averages else self.bin_averages[self.n_averages - 1]
                for val in y_pred.numpy()
            ])
            y_true = np.array([
                self.bin_averages[val]
                if val < self.n_averages else self.bin_averages[self.n_averages - 1]
                for val in y_true.numpy()
            ])
        except:
            y_true = np.array([1])
            y_pred = np.array([1])

        return keras.losses.mse(y_pred, y_true)

    def mixed_categorical_mse(self, y_true, y_pred):
        """_summary_

        Args:
            y_true (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = math.argmax(y_pred, axis=1)
        try:
            y_pred = np.array([
                self.bin_averages[val]
                if val < self.n_averages else self.bin_averages[self.n_averages - 1]
                for val in y_pred.numpy()
            ])
        except:
            pdb.set_trace()

        return keras.losses.mse(y_pred, y_true)
