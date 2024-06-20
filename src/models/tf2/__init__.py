from tensorflow import config
from tensorflow.keras import Model
from utils.IO import *

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
        if metrics is not None:
            for metric in metrics:
                if metric in metric_mapping:
                    metrics[metrics.index(metric)] = metric_mapping[metric]
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
