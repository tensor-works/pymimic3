from models.tf2.mappings import metric_mapping
from tensorflow.keras import Model


class AbstractTf2Model(Model):

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                loss_weights=None,
                metrics=[],
                weighted_metrics=None,
                run_eagerly=False,
                steps_per_execution=1,
                jit_compile='auto',
                auto_scale_loss=True):
        for metric in metrics:
            if metric in metric_mapping:
                metrics[metrics.index(metric)] = metric_mapping[metric]
        super().compile(optimizer=optimizer,
                        loss=loss,
                        loss_weights=loss_weights,
                        metrics=metrics,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        steps_per_execution=steps_per_execution,
                        jit_compile=jit_compile)
