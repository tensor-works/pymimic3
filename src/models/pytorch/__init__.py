import torch
import torch.nn as nn
import numpy as np
import pickle
import warnings
from utils.numeric import RunningAvg
from types import FunctionType
from typing import List, Tuple, Literal
from typing_extensions import Self
from copy import deepcopy
from typing import Union, Dict, overload, Optional
from models.trackers import ModelHistory, LocalModelHistory
from torch.utils.data import DataLoader
from pathlib import Path
from torchmetrics import Metric
from keras.utils import Progbar
from utils import to_snake_case
from torch.optim import Optimizer
from utils import zeropad_samples
from utils.IO import *
from settings import *
from models.pytorch.mappings import *


class AbstractTorchNetwork(nn.Module):

    def __init__(self,
                 final_activation,
                 output_dim: int,
                 model_path: Path = None,
                 task: Literal["multilabel", "multiclass", "binary"] = None,
                 target_repl_coef: float = 0.0):
        super(AbstractTorchNetwork, self).__init__()
        self._model_path = model_path
        self._target_repl_coef = target_repl_coef
        if final_activation is None:
            if output_dim == 1:
                self._final_activation = nn.Sigmoid()
            else:
                self._final_activation = nn.Softmax(dim=-1)
        else:
            self._final_activation = activation_mapping[final_activation]

        if self._model_path is not None:
            # Persistent history
            self._model_path.mkdir(parents=True, exist_ok=True)
            self._history = ModelHistory(Path(self._model_path, "history"))
        else:
            # Mimics the storable
            self._history = LocalModelHistory()

        # This needs to be known for torch metrics
        if task is not None:
            self._task = task
            self._num_classes = output_dim
        elif output_dim == 1:
            self._task = "binary"
            self._num_classes = 1
        elif isinstance(self._final_activation, nn.Softmax):
            self._task = "multiclass"
            self._num_classes = output_dim
        elif isinstance(self._final_activation, nn.Sigmoid):
            self._task = "multilabel"
            self._num_classes = output_dim
        else:
            raise ValueError("Task not specified and could not be inferred from "
                             "output_dim and final_activation! Provide task on initialization.")

        # TODO! this is for debugging remove:
        self._labels = list()
        self._outputs = list()

    def __call__(self, *args, **kwargs):
        raise TypeError(f"'{self.__class__.__name__}' object is not callable")

    def compile(self,
                optimizer: Dict[str, Union[str, type, Optimizer]],
                loss: torch.nn.Module,
                metrics: List[Union[str, type, Metric]] = None,
                class_weight: Union[np.array, np.ndarray] = None,
                clip_value: float = None) -> Self:
        """
        Configures the model for training.

        This function sets up the model with an optimizer, loss function, and evaluation metrics.
        It also supports additional settings such as class weighting and gradient clipping.

        Parameters
        ----------
        optimizer : dict of {str, Union[str, type, Optimizer]}, optional
            Specifies the optimizer to be used during training. It can be a string (key) that refers to
            a pre-mapped optimizer (e.g., 'adam') or an instance of an optimizer class.
            Valid optimizer string values are:
            - sgd
            - adam
            - adadelta
            - adagrad
            - adamax
            - rmsprop
            - nadam 
        loss : torch.nn.Module, optional
            The loss function to minimize during training. This can either be a loss function from
            the loss mapping or a custom `torch.nn.Module`. If a string, the function maps it to
            a predefined loss function (e.g., 'categorical_crossentropy').
            Valid metric string values are:
            - binary_crossentropy
            - logits_binary_crossentropy
            - categorical_crossentropy
            - kld
            - mean_squared_error
            - mean_absolute_error
            - hinge
            - poisson
            - cosine_similarity
            - huber
        metrics : {str, Union[str, type, Metric]}, optional
            A list of metric names or types for model evaluation.
            If not provided, no additional metrics are tracked.
            - accuracy
            - precision
            - recall
            - f1
            - roc_auc
            - msl_error
            - mean_squared_error
            - mean_absolute_error
            - mae
            - mse
            - r2
            - mape
            - confusion_matrix
            - cohen_kappa
            - log_mae
            - custom_mae
            - roc_auc
            - pr_auc
            - micro_roc_auc
            - macro_roc_auc
            - micro_pr_auc
            - macro_pr_auc
        class_weight : numpy.array or numpy.ndarray, optional
            An array representing the weights to be applied to different classes during training. 
            This is used when training imbalanced datasets to emphasize certain classes.
            Default is None.
        clip_value : float, optional
            If provided, enables gradient clipping with the specified value to avoid exploding gradients.
            Default is None.

        Example
        -------
        >>> model.compile(optimizer='adam', loss='categorical_crossentropy', metrics={'accuracy': accuracy_metric})
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._clip_value = clip_value
        self._test_counter = 0

        if isinstance(optimizer, str):
            if optimizer in optimizer_mapping:
                self._optimizer = optimizer_mapping[optimizer](self.parameters(), lr=0.001)
            else:
                raise ValueError(f"Optimizer {optimizer} not supported."
                                 f"Supported optimizers are {optimizer_mapping.keys()}")
        else:
            self._optimizer = optimizer

        if isinstance(loss, str):
            self._apply_activation = loss != "categorical_crossentropy"

            if loss in loss_mapping:
                reduction = "none" if self._target_repl_coef else "mean"

                self._loss = loss_mapping[loss](weight=class_weight, reduction=reduction)
            else:
                raise ValueError(f"Loss {loss} not supported."
                                 f"Supported losses are {loss_mapping.keys()}")
        else:
            self._apply_activation = not isinstance(loss, nn.CrossEntropyLoss)

            if self._target_repl_coef and self._loss.reduction != "none":
                loss = self._loss.__class__(reduction="none")
            self._loss = loss

        self._metrics = self._init_metrics(metrics)

        self.to(self._device)

        return self

    @overload
    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = None) -> List[int]:
        """
        Evaluate the model using input and output arrays.

        This overload is used when the input and output data are provided as NumPy arrays.

        Parameters
        ----------
        x : numpy.ndarray
            Input data in the form of a NumPy array, which the model will use for evaluation.
            Expects dimension (B, T, F) or [(B, T, F), (B, T, 1)] for deep supervision.
        y : numpy.ndarray
            Ground truth labels corresponding to the input data `x`.
            Expects dimensions (B, T, N)
        batch_size : int, optional
            Number of samples per batch during evaluation. If not provided, the entire dataset will be processed at once.

        Returns
        -------
        list
            The evaluation scores of the model based on the provided arrays.
        """
        ...

    @overload
    def evaluate(self, validation_data: DataLoader, batch_size: int = None) -> List[int]:
        """
        Evaluate the model using a DataLoader.

        This overload is used when the validation data is provided as a PyTorch `DataLoader`.

        Parameters
        ----------
        validation_data : DataLoader
            A PyTorch DataLoader object containing the validation dataset, which includes
            batches of input data and corresponding labels. 
            Expects input dimensions  (1, T, F) or [(1, T, F), (1, T, 1)] for deep supervision.
            Expects target dimensions (1, 1, N) or (1, T, N) for deep supervision.
        batch_size : int, optional
            Number of samples per batch during evaluation. If `None`, the batch size defined in the DataLoader will be used.

        Returns
        -------
        list
            The evaluation scores of the model based on the provided arrays.
        """
        ...

    def evaluate(self, *args, **kwargs):

        eval_data, is_array, args, kwargs = self._get_dataloader_or_array(*args, **kwargs)

        if is_array:
            return self._evaluate_with_arrays(*eval_data, is_test=True, *args, **kwargs)

        return self._evaluate_with_dataloader(*eval_data, is_test=True, *args, **kwargs)

    @overload
    def fit(self,
            generator: DataLoader,
            epochs: int = 1,
            patience: Optional[int] = None,
            save_best_only: bool = True,
            restore_best_weights: bool = True,
            sample_weights: Optional[Dict] = None,
            validation_freq: int = 1,
            validation_data: Optional[DataLoader] = None,
            model_path: Optional[Path] = None) -> Dict[str, Dict[int, float]]:
        """
        Fit the model to DataLoader.

        Parameters
        ----------
        generator : DataLoader
            A PyTorch DataLoader that yields batches of training data.
        epochs : int, optional
            Number of epochs to train the model. Defaults to 1.
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
            If None, early stopping is disabled. Defaults to None.
        save_best_only : bool, optional
            If True, only the best model according to validation loss will be saved. Defaults to True.
        restore_best_weights : bool, optional
            Whether to restore model weights from the epoch with the best validation loss. Defaults to True.
        sample_weights : dict, optional
            Optional dictionary specifying sample weights for training data. Defaults to None.
        validation_freq : int, optional
            Frequency (in epochs) at which to evaluate the model on the validation dataset. Defaults to 1.
        validation_data : DataLoader, optional
            A DataLoader object for validation data. Defaults to None.
        model_path : Path, optional
            Path to the directory where model checkpoints will be saved. Defaults to None.

        Returns
        -------
        dict:
            Training history containing the evolution of the loss and metrics.
        """
        ...

    @overload
    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            patience: Optional[int] = None,
            save_best_only: bool = True,
            restore_best_weights: bool = True,
            sample_weights: Optional[Dict] = None,
            validation_freq: int = 1,
            validation_data: Optional[Union[tuple, list]] = None,
            model_path: Optional[Path] = None) -> Dict[str, Dict[int, float]]:
        """
        Fit the model to an array dataset.

        Parameters
        ----------
        x : numpy.ndarray
            Input data for training.
        y : numpy.ndarray
            Ground truth labels for the input data `x`.
        epochs : int, optional
            Number of epochs to train the model. Defaults to 1.
        batch_size : int, optional
            Number of samples per gradient update. Defaults to 32.
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
            If None, early stopping is disabled. Defaults to None.
        save_best_only : bool, optional
            If True, only the best model according to validation loss will be saved. Defaults to True.
        restore_best_weights : bool, optional
            Whether to restore model weights from the epoch with the best validation loss. Defaults to True.
        sample_weights : dict, optional
            Optional dictionary specifying sample weights for training data. Defaults to None.
        validation_freq : int, optional
            Frequency (in epochs) at which to evaluate the model on the validation dataset. Defaults to 1.
        validation_data : Union[tuple, list], optional
            Validation data in the form of a tuple (x_val, y_val) or list. Defaults to None.
        model_path : Path, optional
            Path to the directory where model checkpoints will be saved. Defaults to None.

        Returns
        -------
        dict:
            Training history containing the evolution of the loss and metrics.
        """
        ...

    def fit(self,
            *args,
            batch_size: int = 32,
            epochs: int = 1,
            patience: int = None,
            save_best_only: bool = True,
            restore_best_weights: bool = True,
            sample_weights: dict = None,
            validation_freq: int = 1,
            validation_data: Union[DataLoader, tuple, list] = None,
            model_path: Path = None,
            **kwargs):

        train_data, _, _, _ = self._get_dataloader_or_array(*args, **kwargs)

        if model_path is not None:
            self._model_path = model_path
            self._model_path.mkdir(parents=True, exist_ok=True)

        if validation_data is None and patience is not None:
            warn_io("WARNING:tensorflow:Early stopping conditioned on metric `val_loss` "
                    "which is not available. Available metrics are: loss")

        if patience is None:
            patience = epochs

        self._patience_counter = 0
        initial_epoch = self._latest_epoch(epochs, self._model_path)
        self.load(epochs, weights_only=True)

        for epoch in range(initial_epoch + 1, epochs + 1):

            self._train(*train_data,
                        batch_size=batch_size,
                        epoch=epoch,
                        epochs=epochs,
                        sample_weights=sample_weights,
                        has_val=validation_data is not None)

            if validation_data is not None:
                self.evaluate(validation_data=validation_data,
                              val_frequency=validation_freq,
                              epoch=epoch)

            if validation_data is not None:
                best_epoch = self._history.best_val["epoch"]
            else:
                best_epoch = self._history.best_train["epoch"]

            if self._checkpoint(save_best_only=save_best_only,
                                restore_best_weights=restore_best_weights,
                                epoch=epoch,
                                epochs=epochs,
                                best_epoch=best_epoch,
                                patience=patience):
                break

        return self._history.to_json()

    def load(self, epochs: int, weights_only=False) -> int:
        """
        Load model weights or the full model from the saved checkpoint.

        This method loads the model's state from a checkpoint file saved manually or during training. Depending on 
        the `weights_only` argument, it can either load only the model's parameters or the entire model,
        including attributes such as optimizer and training history.

        Parameters
        ----------
        epochs : int
            The total number of epochs trained. Used to identify the latest checkpoint.
        weights_only : bool, optional
            If True, only the model weights are loaded. If False, the full model including state and attributes
            will be loaded. Defaults to False.

        Returns
        -------
        int
            Returns 1 if the model was successfully loaded, otherwise returns 0.
        """
        if self._model_path is not None:
            latest_epoch = self._latest_epoch(epochs, self._model_path)

            if weights_only:
                checkpoint_path = Path(self._model_path, f"cp-{latest_epoch:04}.ckpt")

                if checkpoint_path.is_file():
                    model_state = torch.load(checkpoint_path)
                    self.load_state_dict(model_state)
                    print(f"Loaded model parameters from {checkpoint_path}")

                    return 1
            else:
                checkpoint_path = Path(self._model_path, f"cp-{latest_epoch:04}.pkl")

                if checkpoint_path.is_file():

                    with open(checkpoint_path, "rb") as load_file:
                        load_params = pickle.load(load_file)

                    for key, value in load_params.items():
                        setattr(self, key, value)

                    print(f"Loaded model from {checkpoint_path}")

                    return 1
        return 0

    @overload
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data using NumPy arrays.
        
        Removes padding, applies masking and manages dimensional conservation when generating predictions.
        To ensure that predictions reflect the models performance, use the predict method over the forward
        method, as preprocessing is applied to the data during training and evaluation, which is not included
        into the forward method.

        Parameters
        ----------
        x : numpy.ndarray
            Input data to be used for generating predictions.
            Expects input dimensions  (B, T, F) or [(B, T, F), (B, T, 1)] for deep supervision.

        Returns
        -------
        numpy.ndarray
            The model's predictions for the provided input data.
            Dimensions (B, 1, N) or (B, T, N) for deep supervision.
        """
        ...

    @overload
    def predict(self, generator: DataLoader) -> np.ndarray:
        """
        Generate predictions using a PyTorch DataLoader.
        
        Removes padding, applies masking and manages dimensional conservation when generating predictions.
        To ensure that predictions reflect the models performance, use the predict method over the forward
        method, as preprocessing is applied to the data during training and evaluation, which is not included
        into the forward method.

        Parameters
        ----------
        generator : DataLoader
            A PyTorch DataLoader providing batches of input data for prediction.
            Expects input dimensions  (1, T, F) or [(1, T, F), (1, T, 1)] for deep supervision.

        Returns
        -------
        numpy.ndarray
            The model's predictions for the provided input data from the DataLoader.
            Dimensions (1, 1, N) or (1, T, N) for deep supervision.
        """
        ...

    def predict(self, *args, batch_size=None, verbose="auto", steps=None, **kwargs):
        predict_data, is_array, args, kwargs = self._get_dataloader_or_array(*args,
                                                                             **kwargs,
                                                                             has_y=False)
        if is_array:
            return self._predict_numpy(*predict_data, batch_size=batch_size, verbose=verbose)

        return self._predict_dataloader(*predict_data, batch_size=batch_size, verbose=verbose)

    def save(self, epoch):
        """
        Save the model's weights at a given epoch.

        This method saves the model's state (parameters) to a checkpoint file.

        Parameters
        ----------
        epoch : int
            The epoch number at which the model is being saved.
        """
        if self._model_path is not None:
            checkpoint_path = Path(self._model_path, f"cp-{epoch:04}.ckpt")
            torch.save(self.state_dict(), checkpoint_path)

    @property
    def optimizer(self) -> Optimizer:
        """
        Return the model's optimizer.

        Returns
        -------
        Optimizer or None
            The optimizer used during training, or None if not defined.
        """
        if hasattr(self, "_optimizer"):
            return self._optimizer

    @property
    def loss(self) -> torch.nn.Module:
        if hasattr(self, "_loss"):
            return self._loss

    @property
    def history(self) -> Union[ModelHistory, LocalModelHistory]:
        return self._history

    def _allowed_key(self, key: str) -> bool:
        return not any([key.endswith(metric) for metric in TEXT_METRICS])

    def _clean_directory(self, best_epoch: int, epochs: int, keep_latest: bool = True):
        """Clean directory of checkpoints, keeping the best epoch and the latest if specified.
        """

        [
            folder.unlink()
            for i in range(epochs + 1)
            for folder in self._model_path.iterdir()
            if f"{i:04d}" in folder.name and ((i != epochs) or not keep_latest) and
            (i != best_epoch) and (".ckpt" in folder.name) and folder.is_file()
        ]

    def _checkpoint(self, save_best_only: bool, restore_best_weights: bool, epoch: int, epochs: int,
                    best_epoch: int, patience: int) -> bool:
        """Save the model and clean the directory of checkpoints. Returns True if patience is exceeded,
           False otherwise.
        """
        if self._model_path is None:
            return False
        if save_best_only and best_epoch == epoch:
            self.save(epoch)
            self._clean_directory(best_epoch, epochs, True)
            self._patience_counter = 0
        elif not save_best_only and self._model_path is not None:
            self.save(epoch)
            self._patience_counter += 1

            if self._patience_counter >= patience:
                if restore_best_weights:
                    self.load_state_dict(
                        torch.load(Path(self._model_path, f"cp-{best_epoch:04}.ckpt")))

                return True

        return False

    def _evaluate_with_arrays(self,
                              x: np.ndarray,
                              y: np.ndarray,
                              batch_size: int = None,
                              val_frequency: int = 1,
                              epoch: int = 0,
                              is_test: bool = False,
                              **kwargs) -> List[float]:
        """Evaluate the model on arrays and return the scores (loss, metrics) as list.
           Can be called from the fit method in which case val_frequency is checked against epoch
           before execution or by you, yes youm the userm you are the real hero, in which case the 
           overload handler will set is_test to True and store results under "test" prefix instead of "val".
        """
        # Evaluation can be on validation or test set
        prefix = "test" if is_test else "val"
        # Don't exectue if not specified or not correct frequency to epoch
        if val_frequency is None:
            return
        elif epoch is None:
            return
        elif not len(x) or not len(y):
            return
        elif (epoch) % val_frequency != 0:
            return

        # Handle masking
        if isinstance(x, (list, tuple)):
            x, masks = x
            masks = torch.tensor(masks, dtype=torch.float32).bool().to(self._device)
            masking_flag = True
        else:
            masking_flag = False

        # Dimension variables
        if len(y.shape) < 3:
            y = np.expand_dims(y, axis=-1)
        x = torch.tensor(x, dtype=torch.float32).to(self._device)
        y = torch.tensor(y).to(self._device)

        # Init counter epoch variables
        self._on_epoch_start(prefix,
                             generator_size=len(y),
                             batch_size=batch_size if batch_size is not None else len(y))

        # Evaluate only if necessary
        self.eval()

        # Batch iter variables
        aggr_outputs = []
        aggr_labels = []

        with torch.no_grad():
            iter_len = len(x) - 1
            for sample_idx, (val_inputs, val_labels) in enumerate(zip(x, y)):
                if masking_flag:
                    # Set labels to zero when masking since forward does the same
                    mask = masks[sample_idx]
                    val_inputs = val_inputs[mask.squeeze(), :]
                    mask = mask.unsqueeze(0)
                else:
                    val_inputs = self._remove_end_padding(val_inputs)
                    mask = None

                # TODO! add target replication here

                # Adjust dimensions
                val_inputs = val_inputs.unsqueeze(0)
                val_labels = val_labels.unsqueeze(0)

                # Create predictions
                val_outputs = self.forward(val_inputs, masks=mask)

                # Apply masking
                if masking_flag:
                    # val_outputs = val_outputs[:, mask.squeeze()]
                    val_labels = val_labels[:, mask.squeeze()]

                # Accumulate outputs and labels either flat or with dim of multilabel
                aggr_outputs.append(val_outputs)
                aggr_labels.append(val_labels)

                aggr_outputs, aggr_labels = self._evaluate_batch(aggr_outputs,
                                                                 aggr_labels,
                                                                 is_test=is_test,
                                                                 finalize=iter_len == sample_idx)

                if sample_idx == iter_len:
                    break

        # Only update history if test
        if is_test:
            self._on_epoch_end(epoch, prefix="test")
            return_metrics = list(dict(self._get_metrics(self._metrics["test"])).values())
            return return_metrics
        # Also complete progbar if eval
        self._on_epoch_end(epoch, prefix="val")
        return_metrics = list(dict(self._get_metrics(self._metrics["val"])).values())
        return return_metrics

    def _evaluate_with_dataloader(self,
                                  generator: DataLoader,
                                  batch_size: int = None,
                                  val_frequency: int = 0,
                                  epoch: int = 0,
                                  is_test: bool = False):
        """
           Evaluate the model on arrays and return the scores (loss, metrics) as list.
           Can be called from the fit method in which case val_frequency is checked against epoch
           before execution or through the external API, in which case the overload handler will 
           set is_test to True and store results under "test" prefix instead of "val".
        """
        # Eval mode
        self.eval()

        # Evaluation can be on validation or test set
        prefix = "test" if is_test else "val"
        # Don't exectue if not specified or not correct frequency to epoch
        if val_frequency is None:
            return
        elif epoch is None:
            return
        elif not len(generator) or not len(generator):
            return
        elif (epoch) % val_frequency != 0:
            return

        # Handle masking
        masking_flag = None

        # Init counter epoch variables
        generator_size = len(generator)
        self._on_epoch_start(prefix,
                             generator_size=generator_size,
                             batch_size=batch_size if batch_size is not None else len(y))

        # Batch iter variables
        aggr_outputs = []
        aggr_labels = []

        # Execute with no grad and eval mode
        self.eval()
        with torch.no_grad():
            # Unroll generator
            iter_len = len(generator) - 1
            for sample_idx, (val_inputs, val_labels) in enumerate(generator):
                if masking_flag == None:
                    # Most efficient way to set this I could think of
                    masking_flag = isinstance(val_input, (list, tuple))

                if masking_flag:
                    val_input, mask = val_input
                    val_input = val_input[mask.squeeze(), :]
                    val_input = val_input.to(self._device)
                    # Set labels to zero when masking since forward does the same
                    mask = mask.to(self._device).bool()
                else:
                    input = input.to(self._device)
                    mask = None

                # Adjust dimensions
                val_inputs = val_inputs.unsqueeze(0)
                val_labels = val_labels.unsqueeze(0)

                # On device label
                val_labels = val_labels.to(self._device)

                # Create predictions
                val_outputs = self.forward(val_inputs, masks=mask)

                # Apply masking
                if masking_flag:
                    # val_outputs = val_outputs[:, mask.squeeze()]
                    val_labels = val_labels[:, mask.squeeze()]

                # Accumulate outputs and labels either flat or with dim of multilabel
                aggr_outputs.append(val_outputs)
                aggr_labels.append(val_labels)

                aggr_outputs, aggr_labels = self._evaluate_batch(aggr_outputs,
                                                                 aggr_labels,
                                                                 is_test=is_test,
                                                                 finalize=iter_len == sample_idx)

                if sample_idx == iter_len:
                    break

        # Only update history if test
        if is_test:
            self._on_epoch_end(epoch, prefix="test")
            return_metrics = list(dict(self._get_metrics(self._metrics["test"])).values())
            return return_metrics
        # Also complete progbar if eval
        self._on_epoch_end(epoch, prefix="val")
        return_metrics = list(dict(self._get_metrics(self._metrics["val"])).values())
        return return_metrics

    def _evaluate_batch(self,
                        outputs: list,
                        labels: list,
                        is_test: bool = False,
                        finalize: bool = False):
        """Evaluates the batch concatenated by higher level _evaluate_with_dataloader or 
           _evaluate_with_arrays method and updates metric state. Finalize will finalize 
           the epoch progbar.
        """
        # This encapsulation creates me a lot of trouble by adding members that are hard to trace
        self._sample_count += 1

        if self._sample_count >= self._batch_size:
            # Concatenate accumulated outputs and labels
            if self._target_repl_coef:
                #
                replication_weights = [
                    torch.cat([
                        torch.ones(output.shape[1] - 1, output.shape[2]) * self._target_repl_coef,
                        torch.ones(1, output.shape[2])
                    ]) for output in outputs
                ]

                replication_weights = torch.cat(replication_weights,
                                                axis=0).to(self._device).squeeze()
                # Replicate along T
                labels = [
                    label.expand(-1, output.shape[1], -1) for label, output in zip(labels, outputs)
                ]

                # Cat along T
                labels = torch.cat(labels, axis=1).squeeze()
                outputs = torch.cat(outputs, axis=1).squeeze()
            else:
                # Cat along T
                outputs = torch.cat(outputs, axis=1).squeeze()

            # If multilabel, labels are one-hot, else they are sparse
            if not self._target_repl_coef:
                if self._task == "multilabel":
                    # T, N
                    labels = torch.stack(labels).squeeze()
                else:
                    # Cat along T then squeeze
                    labels = torch.cat(labels, axis=1).squeeze()

            # Compute loss
            if self._target_repl_coef:
                # Apply target replication loss here
                loss = self._loss(outputs, labels)
                loss = (loss * replication_weights).mean()
            else:
                loss = self._loss(outputs, labels)

            # Reset count
            self._update_metrics(loss,
                                 outputs,
                                 labels,
                                 prefix="test" if is_test else "val",
                                 update_progbar=finalize and not is_test,
                                 finalize=finalize)
            self._sample_count = 0
            self._batch_count += 1

            # Reset aggregator vars
            outputs = []
            labels = []
        return outputs, labels

    def _get_metrics(self, metrics: Dict[str, Metric]) -> List[Tuple[str, float]]:
        """Fetches metrics from the metric state object, moves them to CPU and converts them
           to str(name), float(quant) value tuples.
        """
        keys = list([metric for metric in metrics.keys() if self._allowed_key(metric)])
        values = [metrics[key].compute() for key in keys]
        values = [value.item() if isinstance(value, torch.Tensor) else value for value in values]
        return list(zip(keys, values))

    def _get_dataloader_or_array(self, *args, has_y=True, **kwargs):
        """Decides wether args indicate the use of a dataloader or array and
           return in a structured fashion.
        """
        args = list(args)
        is_array = False
        if len(args) >= 1 and isinstance(args[0], DataLoader):
            data = (args.pop(0),)
        elif "generator" in kwargs:
            data = (kwargs.pop("generator"),)
        else:
            if len(args) >= 2 and isinstance(args[1], (np.ndarray, torch.Tensor)):
                y = deepcopy(args.pop(1))
                is_array = True
            elif "y" in kwargs:
                y = deepcopy(kwargs.pop("y"))
                is_array = True
            elif has_y:
                raise TypeError("Invalid arguments")

            if len(args) >= 1 and isinstance(args[0], (np.ndarray, torch.Tensor, list, tuple)):
                x = deepcopy(args.pop(0))
                is_array = True
            elif "x" in kwargs:
                x = deepcopy(kwargs.pop("x"))
                is_array = True
            else:
                raise TypeError("Invalid arguments")

        if is_array and has_y:
            data = (x, y)
        elif is_array:
            data = (x,)
        return data, is_array, args, kwargs

    def _init_metrics(self,
                      metrics,
                      prefices: list = ["train", "val", "test"]) -> Dict[str, Dict[str, Metric]]:
        """Initialized the metric state for provided prefices. Return nested dictionary of
           prefix -> metric name -> value
        """
        settings = {
            "task": self._task,
            "num_labels" if self._task == "multilabel" else "num_classes": self._num_classes
        }
        return_metrics = {"loss": RunningAvg()}

        # Creat the base metric dict
        for metric in metrics:
            # Map string metrics to objects
            if isinstance(metric, str):
                metric_name = metric
                metric = metric_mapping[metric]
            else:
                # Get the name of object metrics
                try:
                    metric_name = to_snake_case(metric.__name__)
                except:
                    metric_name = "unknonw"
            # If type, instantiate
            if isinstance(metric, (type, FunctionType)):
                metric = metric(**settings)

            return_metrics[metric_name] = metric.to(self._device)

        # metrics -> ""
        #        |-> "train"
        #        |-> "val"
        #        °-> "test"
        # or the specified prefices

        # Create metric dict for each prefix and add to the return dict and history
        prefixed_metrics = dict()
        prefixed_metrics[""] = return_metrics
        for prefix in prefices:
            # Add current prefix
            prefixed_metric = dict()
            for metric_name, metric in return_metrics.items():
                # Ensure deepcopy
                prefixed_metric[metric_name] = deepcopy(metric)

                # Add to history
                metric_history: dict = getattr(self._history, f"{prefix}_metrics")
                # If the metric is not yet in history, then add
                if not metric_name in metric_history:
                    metric_history[metric_name] = dict()

            prefixed_metrics[prefix] = prefixed_metric

        return prefixed_metrics

    def _latest_epoch(self, epochs: int, filepath: Path) -> int:
        """Returns latest saved epoch
        """
        if self._model_path is None:
            return 0
        check_point_epochs = [
            i for i in range(epochs + 1) for folder in filepath.iterdir()
            if f"{i:04d}" in folder.name
        ]

        if check_point_epochs:
            return max(check_point_epochs)

        return 0

    def _on_epoch_start(self,
                        prefix: str,
                        generator_size: int = 0,
                        batch_size: int = 0,
                        has_val: bool = False):
        """Setup epoch state and verbosity before start
        """
        # Insecure about consistency of these here
        self._current_metrics = dict()
        if generator_size:
            self._epoch_progbar = Progbar((generator_size // batch_size))
        self._sample_size = (generator_size // batch_size) * batch_size

        # Batch iter variables
        self._sample_count = 0
        self._batch_count = 1
        self._generator_size = generator_size
        self._batch_size = batch_size
        self._has_val = has_val

        # Reset metric values
        for metric in self._metrics[prefix].values():
            metric.reset()

        self._labels = list()
        self._outputs = list()

    def _on_epoch_end(self, epoch: int, prefix: str = ""):
        """Updates the history and resets counter variables
        """
        # Update loss history
        if hasattr(self._history, f"{prefix}_loss"):
            loss_history = getattr(self._history, f"{prefix}_loss")
            avg_loss = self._current_metrics["loss"]
            if prefix in ["train", "val"]:
                loss_history[epoch] = avg_loss
            else:
                setattr(self._history, f"{prefix}_loss", avg_loss)

        # Update best loss
        if hasattr(self._history, f"best_{prefix}"):
            best_epochs = getattr(self._history, f"best_{prefix}")
            if best_epochs["loss"] > avg_loss:
                best_epochs["loss"] = avg_loss
                best_epochs["epoch"] = epoch

        # Update metric history
        if hasattr(self._history, f"{prefix}_metrics"):
            metric_history = getattr(self._history, f"{prefix}_metrics")
            for key, metric in self._metrics[prefix].items():
                metric_history[key][epoch] = metric.compute()

        self._current_metrics = dict()
        self._sample_count = 0
        self._batch_count = 0
        self._generator_size = 0
        self._batch_size = 0
        self._has_val = False
        '''
        # TODO! remove
        labels = torch.cat(self._labels, axis=1).to("cpu").detach().squeeze().numpy()
        outputs = torch.cat(self._outputs, axis=1).to("cpu").detach().squeeze().numpy()

        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(labels, outputs)
        print(f"PR AUC: {auc(recall, precision)}")
        print(f"ROC AUC: {roc_auc_score(labels, outputs)}")
        '''

    def _optimize_batch(self, outputs: list, labels: list, finalize: bool = False):
        """Applies optimizer to the concatenated outputs and labels, handed down by the higher level
           train_with_dataloader or train_with_array methods and returns updated list of outputs
           and labels. The list is reset to zero if the batch was complete and is unmodified if
           the not enough samples where included to optimize a batch. Finalize influence wether 
           the progbar starts new line or expects evaluation prints.
        """
        # This encapsulation creates me a lot of trouble by adding members that are hard to trace
        self._sample_count += 1

        if self._sample_count >= self._batch_size:
            # Concatenate accumulated outputs and labels
            if self._target_repl_coef:
                # Replication weight tensor
                replication_weights = [
                    torch.cat([
                        torch.ones(output.shape[1] - 1, output.shape[2]) * self._target_repl_coef,
                        torch.ones(1, output.shape[2])
                    ]) for output in outputs
                ]

                replication_weights = torch.cat(replication_weights,
                                                axis=0).to(self._device).squeeze()
                # Replicate targets along T
                labels = [
                    label.expand(-1, output.shape[1], -1) for label, output in zip(labels, outputs)
                ]
                # Cat along T
                labels = torch.cat(labels, axis=1).squeeze()
                outputs = torch.cat(outputs, axis=1).squeeze()
            else:
                # Cat along T
                outputs = torch.cat(outputs, axis=1).squeeze()

            # If multilabel, labels are one-hot, else they are sparse
            if not self._target_repl_coef:
                if self._task == "multilabel":
                    # T, N
                    labels = torch.stack(labels).squeeze()
                else:
                    # Cat along T then squeeze
                    labels = torch.cat(labels, axis=1).squeeze()

            # Compute loss
            if self._target_repl_coef:
                # Apply target replication loss here

                loss = self._loss(outputs, labels)
                loss = (loss * replication_weights).mean()
            else:
                loss = self._loss(outputs, labels)

            # Backward pass and optimization

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if self._clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self._clip_value)

            # Reset count
            self._update_metrics(loss,
                                 outputs,
                                 labels,
                                 prefix="train",
                                 update_progbar=True,
                                 finalize=finalize)
            self._sample_count = 0
            self._batch_count += 1

            # Reset aggregator vars
            outputs = []
            labels = []
        return outputs, labels

    def _prefixed_metrics(self, metrics: List[Tuple[str, float]],
                          prefix: str) -> List[Tuple[str, float]]:
        """Returns the list of (metric, value) pair as list, with the name prefixed.
        """
        if prefix is None:
            return metrics
        return [(f"{prefix}_{metric}", value) for metric, value in metrics]

    def _predict_dataloader(self, generator: DataLoader, **kwargs):
        """ Make predictions on a dataloader while handling masking and dimensions.
        """
        self.eval()

        aggr_outputs = []

        with torch.no_grad():
            for idx, (input, label) in enumerate(generator):
                if masking_flag == None:
                    # Most efficient way to set this I could think of
                    masking_flag = isinstance(input, (list, tuple))

                if masking_flag:
                    input, mask = input
                    input = input.to(self._device)
                    # Set labels to zero when masking since forward does the same
                    mask = mask.to(self._device).bool()
                else:
                    input = input.to(self._device)
                    mask = None

                label = label.to(self._device).T
                output = self.forward(input, masks=mask)

                # Accumulate outputs and labels either flat or with dim of multilabel
                if masking_flag:
                    placeholder = -torch.ones(*mask.shape[:2], output.shape[2])
                    placeholder[:, mask.cpu().squeeze()] = output.cpu()
                    output = placeholder
                aggr_outputs.append(output.cpu().numpy())

        return np.concatenate(aggr_outputs)

    def _predict_numpy(self, x: np.ndarray, **kwargs):
        """ Make predictions on an array while handling masking and dimensions.
        """
        # Some transformations are applied to inputs during prediction, therefore the model should not be
        # called directly
        # Handle masking
        if isinstance(x, (list, tuple)):
            x, masks = x
            masks = torch.tensor(masks, dtype=torch.float32).bool().to(self._device)
            masking_flag = True
        else:
            masking_flag = False

        x = torch.tensor(x, dtype=torch.float32).to(self._device)
        # Evaluate only if necessary
        self.eval()

        # Batch iter variables
        aggr_outputs = []

        with torch.no_grad():
            iter_len = len(x) - 1
            for sample_idx, input in enumerate(x):
                if masking_flag:
                    # Set labels to zero when masking since forward does the same
                    mask = masks[sample_idx]
                    input = input[mask.squeeze(), :]
                    mask = mask.unsqueeze(0)
                else:
                    input = self._remove_end_padding(input)
                    mask = None

                # Adjust dimensions
                input = input.unsqueeze(0)

                # Create predictions
                output = self.forward(input, masks=mask)

                # Accumulate outputs and labels either flat or with dim of multilabel
                if masking_flag:
                    placeholder = -torch.ones(*mask.shape[:2], output.shape[2])
                    placeholder[:, mask.cpu().squeeze()] = output.cpu()
                    output = placeholder
                aggr_outputs.append(output.cpu().numpy())

                if sample_idx == iter_len:
                    break

        return np.concatenate(aggr_outputs)

    def _remove_end_padding(self, tensor) -> torch.Tensor:
        return tensor[:torch.max(torch.nonzero((tensor != 0).long().sum(1))) + 1, :]

    @overload
    def _train(self,
               x: np.ndarray,
               y: np.ndarray,
               batch_size: int,
               epoch: int = 0,
               epochs: int = 1,
               sample_weights: Optional[Dict] = None,
               has_val: bool = False):
        """ Train the network on arrays overload.
        """
        ...

    @overload
    def _train(self,
               generator: DataLoader,
               epoch: int = 0,
               epochs: int = 1,
               sample_weights: Optional[Dict] = None,
               has_val: bool = False):
        """ Train the network on a dataloader.
        """
        ...

    def _train(self, *args, **kwargs):
        """ Overload handler.
        """
        if ((len(args) >= 1 and isinstance(args[0], (np.ndarray, torch.Tensor, list, tuple))) or "x" in kwargs) \
            and (len(args) >= 2 and isinstance(args[1], (np.ndarray, torch.Tensor)) or "y" in kwargs):
            return self._train_with_arrays(*args, **kwargs)
        elif (len(args) >= 1 and isinstance(args[0], DataLoader) or "generator" in kwargs):
            return self._train_with_dataloader(*args, **kwargs)
        else:
            raise TypeError("Invalid arguments")

    def _train_with_arrays(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           batch_size: int,
                           epoch: int = 0,
                           epochs: int = 1,
                           sample_weights: dict = None,
                           has_val: bool = False):

        print(f'\nEpoch {epoch}/{epochs}')
        self.train()

        # Handle masking
        if isinstance(x, (list, tuple)):
            x, masks = x
            masks = torch.tensor(masks, dtype=torch.float32).bool().to(self._device)
            masking_flag = True
        else:
            masking_flag = False

        # Dimension variables
        data_size = x.shape[0]

        # Shuffled data
        idx = np.random.permutation(len(x))
        x = torch.tensor(x[idx, :, :], dtype=torch.float32).to(self._device)
        y = torch.tensor(y[idx, :, :]).to(self._device)

        # Init counter epoch variables
        self._on_epoch_start("train", data_size, batch_size, has_val)

        # Batch iter variables
        aggr_outputs = []
        aggr_labels = []

        # Main loop
        iter_len = (len(x) // batch_size) * batch_size - 1
        for sample_idx, (input, label) in enumerate(zip(x, y)):
            label: torch.Tensor
            if masking_flag:
                # Set labels to zero when masking since forward does the same
                mask = masks[sample_idx]
                input = input[mask.squeeze(), :]
                mask = mask.unsqueeze(0)
            else:
                # TODO! Is there a better way. What if there is an actual patient with all zero across
                # TODO! 59 columns? Am I paranoid. Maybe adding the discretizer masking can alleviat
                input = self._remove_end_padding(input)
                mask = None

            if self._target_repl_coef:
                mask = torch.ones(1, input.shape[1], 1)

            # Adjust dimensions
            input = input.unsqueeze(0)
            label = label.unsqueeze(0)

            # Create predictions
            output = self.forward(input, masks=mask)

            if masking_flag:
                # Apply mask to T (B, T, N)
                # output = output[:, mask.squeeze()]
                label = label[:, mask.squeeze()]

            # Accumulate outputs and labels either flat or with dim of multilabel
            aggr_outputs.append(output)
            aggr_labels.append(label)

            # Optimizer network on abtch
            self._labels.append(label)
            self._outputs.append(output)

            aggr_outputs, \
            aggr_labels = self._optimize_batch(outputs=aggr_outputs, \
                                               labels=aggr_labels, \
                                               finalize=(not has_val and sample_idx == iter_len))

            # Miss inclomplete batch
            if sample_idx == iter_len:
                break

        self._on_epoch_end(epoch, prefix="train")

    def _train_with_dataloader(self,
                               generator: DataLoader,
                               batch_size: int,
                               epoch: int = 0,
                               epochs: int = 1,
                               sample_weights: dict = None,
                               has_val: bool = False):
        # Epoch verbosity
        print(f'\nEpoch {epoch}/{epochs}')

        # Train mode
        self.train()
        masking_flag = None

        # Tracking variables
        generator_size = len(generator)
        self._on_epoch_start("train",
                             generator_size=generator_size,
                             batch_size=batch_size,
                             has_val=has_val)
        aggr_outputs = []
        aggr_labels = []

        # Main loop
        iter_len = (len(generator) // batch_size) * batch_size - 1
        for idx, (input, label) in enumerate(generator):
            input: torch.Tensor
            label: torch.Tensor
            if masking_flag == None:
                # Most efficient way to set this I could think of
                masking_flag = isinstance(input, (list, tuple))

            if masking_flag:
                input, mask = input
                input = input[mask.squeeze(), :]
                input = input.to(self._device)
                # Set labels to zero when masking since forward does the same
                mask = mask.to(self._device).bool()
            else:
                input = input.to(self._device)
                mask = None

            label = label.unsqueeze(0)
            input = input.unsqueeze(0)

            # On device label
            label = label.to(self._device)

            # Prediction
            output = self.forward(input, masks=mask)

            if masking_flag:
                # Apply mask to T (B, T, N)
                # output = output[:, mask.squeeze()]
                label = label[:, mask.squeeze()]

            # Accumulate outputs and labels
            aggr_outputs.append(output)
            aggr_labels.append(label)

            # Optimizer network on abtch
            aggr_outputs, \
            aggr_labels = self._optimize_batch(outputs=aggr_outputs, \
                                               labels=aggr_labels, \
                                               finalize=not has_val and iter_len == idx)

            # Miss inclomplete batch
            if idx == self._sample_size:
                break

        self._on_epoch_end(epoch, prefix="train")

    def _update_metrics(self,
                        loss: torch.Tensor,
                        outputs: torch.Tensor,
                        labels: torch.Tensor,
                        prefix: str,
                        finalize: bool = False,
                        update_progbar: bool = False):
        """ Update the loss running average and the metrics base on the outputs and labels for 
            the given prefix.
        """
        # Update metrics
        with torch.no_grad():
            # https://torchmetrics.readthedocs.io/en/v0.8.0/pages/classification.html
            # What do we need?
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        category=UserWarning,
                                        module="torchmetrics.utilities.prints")

                for name, metric in self._metrics[prefix].items():
                    if name == "loss":
                        metric.update(loss.item())
                    else:
                        metric.update(
                            outputs,
                            labels.int() if hasattr(labels, "int") else labels.astype(int))

                # Reset accumulators and count
                self._current_metrics["loss"] = loss.item()
                self._current_metrics.update(dict(self._get_metrics(self._metrics[prefix])))

                if update_progbar:
                    self._epoch_progbar.update(
                        self._epoch_progbar.target if prefix == "val" else self._batch_count,
                        values=self._prefixed_metrics(self._get_metrics(self._metrics[prefix]),
                                                      prefix=prefix if prefix != "train" else None),
                        finalize=finalize)  # and self._batch_count == self._epoch_progbar.target)
