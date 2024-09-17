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
    """
    Extended TF2 model class for the MIMIC-III dataset. The model allows for the specification of alternative
    metrics as string, such as "roc_auc" and "roc_pr". In addition, the class handels deep supervision and
    masking for the implemented models.
    """

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
        """
        Configures the model for training.

        This method allows specifying the optimizer, loss function, and metrics for the model. 

        Parameters
        ----------
        optimizer : str or `Optimizer` instance, optional
            The optimizer to use for training. Default is 'rmsprop'.
        
        loss : str or `Loss` instance, optional
            The loss function to use. Default is None.

        metrics : list, optional
            A list of metrics to evaluate during training. Default is None.
            If not provided, no additional metrics are tracked.
            Valid metrics strings are: 
            - roc_auc
            - micro_roc_auc
            - macro_roc_auc
            - pr_auc
            - cohen_kappa
            - log_mae
            - custom_mae
            - accuracy
            - acc
            - binary_accuracy
            - categorical_accuracy
            - sparse_categorical_accuracy
            - AUC
            - precision
            - recall
            - specificity_at_sensitivity
            - sensitivity_at_specificity
            - hinge
            - squared_hinge
            - top_k_categorical_accuracy
            - sparse_top_k_categorical_accuracy
            - mean_absolute_error
            - mae
            - mean_squared_error
            - mse
            - mean_absolute_percentage_error
            - mape
            - mean_squared_logarithmic_error
            - msle
            - huber
            - logcosh
            - binary_crossentropy
            - categorical_crossentropy
            - sparse_categorical_crossentropy
            - cosine_proximity
            - cosine_similarity

        loss_weights : list or dict, optional
            Optional list or dictionary specifying scalar coefficients to weight the loss contributions of 
            different model outputs. Default is None.

        weighted_metrics : list, optional
            List of metrics to evaluate on weighted inputs. Default is None.

        run_eagerly : bool, optional
            If True, this model will run eagerly. Default is None.

        steps_per_execution : int, optional
            The number of steps to run before updating the weights in the backpropagation. Default is None.

        jit_compile : bool, optional
            If True, compiles the model using XLA (Accelerated Linear Algebra) for optimization. Default is None.

        pss_evaluation_shards : int, optional
            Parameter related to evaluation on multiple shards. Default is 0.

        kwargs : dict
            Additional keyword arguments.
        """
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

        # Not well understood but this causes problems
        self._metrics = list()

    @tf.function
    def train_step(self, data):
        """ Modified train step to handle deep supervision and masking, with dynamic reassignment
            of metrics to weighted metrics (to apply masking) if deep supervision detected. 
        """
        # Get the data tf2 style
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Resolve for deep supervision
        if not hasattr(self, '_deep_supervision'):
            if isinstance(x, (list, tuple)):
                # Check for masking once for performance
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
        # Handle deep supervision if already resolved
        elif self._deep_supervision:
            mask = x[-1]
        else:
            mask = None

        # Apply masking to sample weights
        if sample_weight is not None:
            sample_weight = sample_weight * mask
        elif mask is not None:
            sample_weight = tf.cast(mask, dtype=tf.float32)

        # Compute loss
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            if not self._deep_supervision:
                y = tf.squeeze(y, axis=1)

            loss = self.compiled_loss(y,
                                      y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        # Update network
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """ Modified evaluation step to handle deep supervision and masking, with dynamic reassignment
            of metrics to weighted metrics (to apply masking) if deep supervision detected. 
        """
        # Get the data tf2 style
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Resolve for deep supervision
        if not hasattr(self, '_deep_supervision'):
            if isinstance(x, (list, tuple)):
                # Check for masking once for performance
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
        # Handle deep supervision if already resolved
        elif self._deep_supervision:
            mask = x[-1]
        else:
            mask = None

        # Apply masking to sample weights
        if sample_weight is not None:
            sample_weight = sample_weight * tf.cast(mask, dtype=tf.float32)
        elif mask is not None:
            sample_weight = tf.cast(mask, dtype=tf.float32)

        # Make prediction
        y_pred = self(x, training=False)
        if not self._deep_supervision:
            y = tf.squeeze(y, axis=1)

        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)
