from copy import deepcopy
from keras.src.engine import compile_utils
from tensorflow import config
from keras.api._v2.keras import Model
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training
from utils.IO import *
from utils.arrays import isiterable

import tensorflow as tf

try:
    gpus = config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        config.experimental.set_memory_growth(gpu, True)
except:
    warn_io("Could not set dynamic memory growth for GPUs. This may lead to memory errors.")

from models.tf2.mappings import metric_mapping


class AbstractTf2Model(Model):

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                jit_compile=None,
                pss_evaluation_shards=0,
                **kwargs):
        metrics = deepcopy(metrics)
        weighted_metrics = deepcopy(weighted_metrics)
        if metrics is not None:
            for metric in metrics:
                if metric in metric_mapping:
                    metrics[metrics.index(metric)] = metric_mapping[metric]
        if weighted_metrics is not None:
            for metric in weighted_metrics:
                if metric in metric_mapping:
                    weighted_metrics[weighted_metrics.index(metric)] = metric_mapping[metric]

        self._combined_metrics_buffer = (metrics or []) + (weighted_metrics or []) or None
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        steps_per_execution=steps_per_execution,
                        jit_compile=jit_compile,
                        pss_evaluation_shards=pss_evaluation_shards,
                        **kwargs)

    @tf.function
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if not hasattr(self, '_deep_supervision'):
            # Check for masking once for performance
            if isinstance(x, (list, tuple)):
                mask = x[-1]
                self._deep_supervision = True
                self._metrics = []
                self.compiled_metrics = compile_utils.MetricsContainer(
                    None,
                    self._combined_metrics_buffer,
                    output_names=self.output_names,
                )
            else:
                mask = None
                self._deep_supervision = False
        elif self._deep_supervision:
            mask = x[-1]
        else:
            mask = None

        if sample_weight is not None:
            sample_weight = sample_weight * mask
        elif mask is not None:
            sample_weight = tf.cast(mask, dtype=tf.float32)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            if not self._deep_supervision:
                y = tf.squeeze(y, axis=1)

            loss = self.compiled_loss(y,
                                      y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if not hasattr(self, '_deep_supervision'):
            # Check for masking once for performance
            if isinstance(x, (list, tuple)):
                mask = x[-1]
                self._deep_supervision = True
                self._metrics = []
                self.compiled_metrics = compile_utils.MetricsContainer(
                    None,
                    self._combined_metrics_buffer,
                    output_names=self.output_names,
                )
            else:
                mask = None
                self._deep_supervision = False
        elif self._deep_supervision:
            mask = x[-1]
        else:
            mask = None

        if sample_weight is not None:
            sample_weight = sample_weight * tf.cast(mask, dtype=tf.float32)
        elif mask is not None:
            sample_weight = tf.cast(mask, dtype=tf.float32)

        y_pred = self(x, training=False)
        if not self._deep_supervision:
            y = tf.squeeze(y, axis=1)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)
