#
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from managers import ReducedCheckpointManager


class IncrementalLogReg(object):

    def __init__(self,
                 task: str,
                 input_dim: int = 714,
                 alpha: float = 0.001,
                 max_iter: int = 1000,
                 output_dim: int = None,
                 random_state: int = 42) -> None:

        self.max_iter = max_iter
        # TODO! not realy happy with this solution
        self.alpha = alpha
        activation = {
            "DECOMP": 'sigmoid',
            "IHM": 'sigmoid',
            "LOS": 'softmax',
            "PHENO": 'softmax',
            None: None if output_dim == 1 else 'softmax'
        }

        num_classes = {"DECOMP": 1, "IHM": 1, "LOS": 1, "PHENO": 25, None: output_dim}

        self._num_outputs = num_classes[task]

        input = layers.Input(shape=(input_dim), name='x')
        x = layers.Dense(self._num_outputs, activation=activation[task], input_dim=input_dim)(input)
        self._model = Model(inputs=input, outputs=x)
        self._output_tensor = x

    def load(self, checkpoint_folder):
        manager = ReducedCheckpointManager(checkpoint_folder)

        if not manager.is_empty():
            self._model = manager.load_model()

        return

    def latest_epoch(self, checkpoint_folder):
        manager = ReducedCheckpointManager(checkpoint_folder)
        return manager.latest_epoch()

    def __getattr__(self, name: str):
        """ Surrogate to the _model attributes internals.

        Args:
            name (str): name of the method/attribute

        Returns:
            any: method/attribute of _model
        """
        if name in ["load", "_model", "compile"]:
            return self.__getattribute__(name)
        return getattr(self._model, name)

    def compile(self,
                optimizer="rmsprop",
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                jit_compile=None,
                **kwargs):
        """_summary_

        Args:
            optimizer (str, optional): _description_. Defaults to "rmsprop".
            metrics (_type_, optional): _description_. Defaults to None.
            loss_weights (_type_, optional): _description_. Defaults to None.
            weighted_metrics (_type_, optional): _description_. Defaults to None.
            run_eagerly (_type_, optional): _description_. Defaults to None.
            steps_per_execution (_type_, optional): _description_. Defaults to None.
            jit_compile (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if isinstance(optimizer, str):
            optimizer_switch = {"rmsprop": RMSprop(self.alpha), "adam": Adam(self.alpha)}
            optimizer = optimizer_switch[optimizer]
        else:
            optimizer.learning_rate = self.alpha
        if self._num_outputs == 1:
            loss = "binary_crossentropy"
        else:
            loss = "categorical_crossentropy"
        return self._model.compile(optimizer=optimizer,
                                   loss=loss,
                                   metrics=metrics,
                                   loss_weights=loss_weights,
                                   weighted_metrics=weighted_metrics,
                                   run_eagerly=run_eagerly,
                                   steps_per_execution=steps_per_execution,
                                   **kwargs)
