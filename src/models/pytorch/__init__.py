import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import warnings
from types import FunctionType
from typing import List, Tuple
from copy import deepcopy
from typing import Union, Dict, overload, Optional
from models.trackers import ModelHistory, LocalModelHistory
from torch.utils.data import DataLoader
from pathlib import Path
from torchmetrics import Metric
from keras.utils import Progbar
from utils import to_snake_case
from torch.optim import Optimizer
from utils.IO import *
from settings import *
from models.pytorch.mappings import *


class AbstractTorchNetwork(nn.Module):

    def __init__(self, final_activation, output_dim: int, model_path: Path = None):
        super(AbstractTorchNetwork, self).__init__()
        self._model_path = model_path
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
        if output_dim == 1:
            self._task = "binary"
            self._num_classes = 1
        elif isinstance(self._final_activation, nn.Softmax):
            self._task = "multiclass"
            self._num_classes = output_dim
        elif isinstance(self._final_activation, nn.Sigmoid):
            self._task = "multilabel"
            self._num_classes = output_dim

    @property
    def optimizer(self):
        if hasattr(self, "_optimizer"):
            return self._optimizer

    @property
    def loss(self):
        if hasattr(self, "_loss"):
            return self._loss

    @property
    def history(self):
        return self._history

    def save(self, epoch):
        """_summary_
        """
        if self._model_path is not None:
            checkpoint_path = Path(self._model_path, f"cp-{epoch:04}.ckpt")
            torch.save(self.state_dict(), checkpoint_path)

    def load(self, epochs: int, weights_only=False):
        """_summary_

        Returns:
            _type_: _description_
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

    def _allowed_key(self, key: str):
        return not any([key.endswith(metric) for metric in TEXT_METRICS])

    def _clean_directory(self, best_epoch: int, epochs: int, keep_latest: bool = True):
        """_summary_

        Args:
            best_epoch (int): _description_
            keep_latest (bool, optional): _description_. Defaults to True.
        """

        [
            folder.unlink()
            for i in range(epochs + 1)
            for folder in self._model_path.iterdir()
            if f"{i:04d}" in folder.name and ((i != epochs) or not keep_latest) and
            (i != best_epoch) and (".ckpt" in folder.name) and folder.is_file()
        ]

    def _get_metrics(self, metrics: Dict[str, Metric]) -> List[Tuple[str, float]]:
        keys = list([metric for metric in metrics.keys() if self._allowed_key(metric)])
        values = [metrics[key]["value"] for key in keys]
        return list(zip(keys, values))

    def _init_metrics(self,
                      metrics,
                      prefices: list = ["train", "val", "test"]) -> Dict[str, Metric]:
        settings = {"task": self._task, "num_labels": self._num_classes}
        return_metrics = {"loss": {"obj": None, "value": 0.0}}

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

            return_metrics[metric_name] = {"obj": metric.to(self._device), "value": 0.0}

        # Create metric dict for each prefix and add to the return dict
        prefixed_metrics = dict()
        prefixed_metrics[""] = return_metrics
        for prefix in prefices:
            prefixed_metric = dict()
            for metric_name, metric in return_metrics.items():
                prefixed_metric[metric_name] = deepcopy(metric)
                metric_history: dict = getattr(self._history, f"{prefix}_metrics")
                # If the metric is not yet in history, then add
                if not metric_name in metric_history:
                    metric_history[metric_name] = dict()

            prefixed_metrics[prefix] = prefixed_metric

        return prefixed_metrics

    def compile(self,
                optimizer: Dict[str, Union[str, type, Optimizer]] = None,
                loss=None,
                metrics: Dict[str, Union[str, type, Metric]] = None,
                class_weight=None,
                clip_value: float = None):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._clip_value = clip_value
        self._test_counter = 0

        if isinstance(optimizer, str):
            if optimizer in optimizer_mapping:
                self._optimizer = optimizer_mapping[optimizer](self.parameters(), lr=0.001)
            else:
                raise ValueError(f"Optimizer {optimizer} not supported."
                                 f"Supported optimizers are {optimizer_mapping.keys()}")
        elif optimizer is None:
            self._optimizer = optim.Adam(self.parameters(), lr=0.001)
        else:
            self._optimizer = optimizer

        if isinstance(loss, str):
            if loss in loss_mapping:
                self._loss = loss_mapping[loss](weight=class_weight)
            else:
                raise ValueError(f"Loss {loss} not supported."
                                 f"Supported losses are {loss_mapping.keys()}")
        else:
            self._loss = loss
        self._metrics = self._init_metrics(metrics)

        self.to(self._device)

    def _latest_epoch(self, epochs: int, filepath: Path):
        """_summary_

        Returns:
            _type_: _description_
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

    @overload
    def _train(self,
               x: np.ndarray,
               y: np.ndarray,
               batch_size: int,
               epoch: int = 0,
               epochs: int = 1,
               sample_weights: Optional[Dict] = None,
               has_val: bool = False):
        ...

    @overload
    def _train(self,
               generator: DataLoader,
               epoch: int = 0,
               epochs: int = 1,
               sample_weights: Optional[Dict] = None,
               has_val: bool = False):
        ...

    def _train(self, *args, **kwargs):
        if ((len(args) >= 1 and isinstance(args[0], (np.ndarray, list, tuple))) or "x" in kwargs) \
            and (len(args) >= 2 and isinstance(args[1], np.ndarray) or "y" in kwargs):
            return self._train_with_arrays(*args, **kwargs)
        elif (len(args) >= 1 and isinstance(args[0], DataLoader) or "generator" in kwargs):
            return self._train_with_dataloader(*args, **kwargs)
        else:
            raise TypeError("Invalid arguments")

    def _on_epoch_start(self, generator_size: int = 0, batch_size: int = 0, has_val: bool = False):
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

    def _on_epoch_end(self, epoch: int, prefix: str = ""):
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

        if hasattr(self._history, f"{prefix}_metrics"):
            metric_history = getattr(self._history, f"{prefix}_metrics")
            for key, metric in self._metrics[prefix].items():
                metric_history[key][epoch] = metric["value"]

        self._current_metrics = dict()
        self._sample_count = 0
        self._batch_count = 0
        self._generator_size = 0
        self._batch_size = 0
        self._has_val = False

    def _update_metrics(self,
                        loss: torch.Tensor,
                        outputs: torch.Tensor,
                        labels: torch.Tensor,
                        prefix: str,
                        finalize: bool = False,
                        update_progbar: bool = False):
        # Update metrics
        with torch.no_grad():
            # https://torchmetrics.readthedocs.io/en/v0.8.0/pages/classification.html
            # What do we need?
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=
                    "No positive samples in targets, true positive value should be meaningless. "
                    "Returning zero tensor in true positive score")
                for name, metric in self._metrics[prefix].items():
                    if name == "loss":
                        runing_avg = metric["value"] + (loss.item() -
                                                        metric["value"]) / (self._batch_count)
                        # Extra step necessary for actually updating the value
                        metric.update({"value": runing_avg})
                    else:
                        metric["obj"].update(
                            outputs,
                            labels.int() if hasattr(labels, "int") else labels.astype(int))
                        new_value = metric["obj"].compute().item()
                        runing_avg = metric["value"] + (new_value -
                                                        metric["value"]) / (self._batch_count)
                        # Extra step necessary for actually updating the value
                        metric.update({"value": runing_avg})
        # Reset accumulators and count
        self._current_metrics["loss"] = loss.item()
        self._current_metrics.update(dict(self._get_metrics(self._metrics[prefix])))

        if update_progbar:
            self._epoch_progbar.update(
                self._epoch_progbar.target if prefix == "val" else self._batch_count,
                values=self._prefixed_metrics(self._get_metrics(self._metrics[prefix]),
                                              prefix=prefix if prefix != "train" else None),
                finalize=finalize)

    def _prefixed_metrics(self, metrics: List[Tuple[str, float]],
                          prefix: str) -> List[Tuple[str, float]]:
        if prefix is None:
            return metrics
        return [(f"{prefix}_{metric}", value) for metric, value in metrics]

    def remove_end_padding(self, tensor):
        return tensor[:torch.max(torch.nonzero((tensor != 0).long().sum(1))) + 1, :]

    def _train_with_arrays(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           batch_size: int,
                           epoch: int = 0,
                           epochs: int = 1,
                           sample_weights: dict = None,
                           has_val: bool = False):
        # TODO! while keeping the generator at single sample maybe using padded squences from torch ist still
        # TODO! more efficient
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
        idx = np.random.permutation(len(x))
        x = torch.tensor(x[idx, :, :], dtype=torch.float32).to(self._device)
        y = torch.tensor(y[idx, :, :]).to(self._device)

        self._on_epoch_start(data_size, batch_size, has_val)

        # Batch iter variables
        aggr_outputs = []
        aggr_labels = []

        # Main loop
        iter_len = len(x) - 1
        for sample_idx, (input, label) in enumerate(zip(x, y)):
            if masking_flag:
                # Set labels to zero when masking since forward does the same
                mask = masks[sample_idx]
                if len(input.shape) < 3:
                    mask = mask.unsqueeze(0)
                mask = mask
            else:
                # TODO! Is there a better way. What if there is an actual patient with all zero across
                # TODO! 59 columns? Am I paranoid. Maybe adding the discretizer masking can alleviat
                input = self.remove_end_padding(input)
                mask = None
            if len(input.shape) < 3:
                input = input.unsqueeze(0)

            # if len(label.shape) < 3:
            #     label = label.unsqueeze(0)

            # labels = labels * masks
            output = self(input, masks=mask)
            if masking_flag:
                output = torch.masked_select(output, mask)
                label = torch.masked_select(label, mask)
            # Accumulate outputs and labels either flat or with dim of multilabel
            aggr_outputs.append(output)
            aggr_labels.append(label)
            #aggr_outputs.append(
            #    output.view(*(-1,) if self._num_classes == 1 else (-1, self._num_classes),))
            #aggr_labels.append(
            #    label.view(
            #        *(-1,) if self._num_classes == 1 or self._final_activation == "softmax" else
            #        (-1, self._num_classes),))

            # Optimizer network on abtch
            aggr_outputs, \
            aggr_labels = self._optimize_batch(outputs=aggr_outputs, labels=aggr_labels, finalize=(not has_val and sample_idx == iter_len))

            # Miss inclomplete batch
            if sample_idx == self._sample_size:
                break

        self._on_epoch_end(epoch, prefix="train")

    def _train_with_dataloader(self,
                               generator: DataLoader,
                               batch_size: int,
                               epoch: int = 0,
                               epochs: int = 1,
                               sample_weights: dict = None,
                               has_val: bool = False):
        print(f'\nEpoch {epoch}/{epochs}')
        self.train()
        masking_flag = None

        #Tracking variables
        generator_size = len(generator)
        self._on_epoch_start(generator_size, batch_size, has_val)
        aggr_outputs = []
        aggr_labels = []

        # Main loop
        iter_len = len(generator) - 1
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
            label = label.to(self._device)

            # labels = labels * masks
            output = self(input, masks=mask)
            if masking_flag:
                output = torch.masked_select(output, mask)
                label = torch.masked_select(label, mask)
            # Accumulate outputs and labels
            aggr_outputs.append(output.view(-1, self._num_classes))
            aggr_labels.append(label.view(-1))

            # Optimizer network on abtch
            aggr_outputs, \
            aggr_labels = self._optimize_batch(outputs=aggr_outputs, labels=aggr_labels,finalize=not has_val and iter_len == idx)

            # Miss inclomplete batch
            if idx == self._sample_size:
                break

        self._on_epoch_end(epoch, prefix="train")

    def _optimize_batch(self, outputs: list, labels: list, finalize: bool = False):
        # This encapsulation creates me a lot of trouble by adding members that are hard to trace
        self._sample_count += 1

        if self._sample_count >= self._batch_size:
            # Concatenate accumulated outputs and labels
            outputs = torch.cat(outputs)

            # If multilabel, labels are one-hot, else they are sparse
            if self._task == "multilabel":
                labels = torch.cat(labels, axis=1).T
            else:
                labels = torch.cat(labels)

            # Compute loss
            loss = self._loss(outputs, labels)

            # Backward pass and optimization
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if self._clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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

    def _evaluate_with_arrays(self,
                              x: np.ndarray,
                              y: np.ndarray,
                              val_frequency: int,
                              epoch: int,
                              is_test: bool = False):
        prefix = "test" if is_test else "val"
        self._on_epoch_start()
        if isinstance(x, (list, tuple)):
            x, y = x
            masking_flag = True
        else:
            masking_flag = False
        # Evaluate only if necessary
        if (val_frequency is not  None and epoch is not None) \
            and len(x) and len(y) and (epoch) % val_frequency == 0:
            self.eval()
            val_losses = []
            with torch.no_grad():
                x = torch.tensor(x, dtype=torch.float32).to(self._device)
                y = torch.tensor(y, dtype=torch.float32).to(self._device)
                last_iter = len(x) - 1
                for idx, (val_inputs, val_labels) in enumerate(zip(x, y)):
                    if masking_flag:
                        val_inputs, val_masks = val_inputs
                        if len(val_masks.shape) < 3:
                            val_masks: torch.Tensor = val_masks.unsqueeze(0)
                    else:
                        val_masks = None

                    if len(val_inputs.shape) < 3:
                        val_inputs: torch.Tensor = val_inputs.unsqueeze(0)
                    if len(val_labels.shape) < 3:
                        val_labels: torch.Tensor = val_labels.unsqueeze(0)

                    if masking_flag:
                        val_outputs = torch.masked_select(val_outputs, val_masks)
                        val_labels = torch.masked_select(val_labels, val_masks)

                    val_outputs = self(val_inputs, masks=val_masks)
                    val_loss = self._loss(val_outputs.view(-1), val_labels.view(-1))
                    val_losses.append(val_loss.item())
                    self._update_metrics(val_loss,
                                         val_outputs,
                                         val_labels,
                                         prefix=prefix,
                                         update_progbar=(idx == last_iter) and prefix == "val")
                    self._batch_count += 1

            avg_val_loss = np.mean(val_losses)
            # Only update history if test
            if is_test:
                self._on_epoch_end(epoch, prefix="test")
            # Also complete progbar if eval
            else:
                self._on_epoch_end(epoch, prefix="val")
            return avg_val_loss

    def _evaluate_with_generators(self,
                                  generator: DataLoader,
                                  val_frequency: int = 0,
                                  epoch: int = 0,
                                  is_test: bool = False):
        prefix = "test" if is_test else "val"
        self._on_epoch_start()
        if generator is not None and (not val_frequency or (epoch) % val_frequency == 0):
            masking_flag = None
            val_losses = []

            # Execute with no grad and eval mode
            self.eval()
            with torch.no_grad():
                # Unroll generator
                last_iter = len(generator) - 1
                for idx, (val_inputs, val_labels) in enumerate(generator):
                    if masking_flag == None:
                        # Most efficient way to set this I could think of
                        masking_flag = isinstance(val_inputs, (list, tuple))
                    if masking_flag:
                        val_inputs, val_masks = val_inputs
                        val_inputs = val_inputs.to(self._device)
                        val_masks = val_masks.to(self._device)
                    else:
                        val_inputs = val_inputs.to(self._device)
                        val_masks = None
                    val_labels = val_labels.to(self._device)
                    val_outputs = self(val_inputs, masks=val_masks)
                    val_loss = self._loss(val_outputs, val_labels)
                    val_losses.append(val_loss.item())
                    self._update_metrics(val_loss,
                                         val_outputs,
                                         val_labels,
                                         prefix=prefix,
                                         update_progbar=(idx == last_iter) and prefix == "val")
                    self._batch_count += 1

            avg_val_loss = np.mean(val_losses)
            if is_test:
                self._test_counter += 1
                self._on_epoch_end(self._test_counter, prefix="test")
            else:
                self._on_epoch_end(epoch, prefix="val")
            return avg_val_loss

    @overload
    def evaluate(self, x: np.ndarray, y: np.ndarray):
        ...

    @overload
    def evaluate(self, validation_data: DataLoader):
        ...

    def evaluate(self, *args, **kwargs):
        return self._evaluate(*args, is_test=True, epoch=None, val_frequency=None, **kwargs)

    def _evaluate(self, *args, **kwargs):
        # Array evaluation
        if (len(args) >= 1 and isinstance(args[0], (list, tuple))):
            # Unpack and pass individual
            x, y = args[0]
            return self._evaluate_with_arrays(x, y, *args, **kwargs)
        elif "validation_data" in kwargs and isinstance(kwargs["validation_data"], (list, tuple)):
            x, y = kwargs.pop("validation_data")
            return self._evaluate_with_arrays(x, y, *args, **kwargs)
        # Generator evaluation
        elif len(args) >= 1 and isinstance(args[0], DataLoader):
            return self._evaluate_with_generators(*args, **kwargs)
        elif "validation_data" in kwargs and isinstance(kwargs["validation_data"], DataLoader):
            kwargs["generator"] = kwargs.pop("validation_data")
            return self._evaluate_with_generators(*args, **kwargs)
        else:
            if len(args) >= 1 and isinstance(args[0], np.ndarray):
                x = args[0]
            elif "x" in kwargs:
                x = kwargs["x"]

            if len(args) >= 2 and isinstance(args[1], np.ndarray):
                y = args[1]
            elif "y" in kwargs:
                y = kwargs["y"]
            try:
                return self._evaluate_with_arrays(x, y, **kwargs)
            except TypeError as e:
                raise TypeError("Invalid arguments")

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
            model_path: Optional[Path] = None):
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
            model_path: Optional[Path] = None):
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
        is_array = False
        if len(args) >= 1 and isinstance(args[0], DataLoader):
            train_data = (args[0],)
        elif "generator" in kwargs:
            train_data = (kwargs["generator"],)
        else:
            if len(args) >= 1 and isinstance(args[0], (np.ndarray, torch.Tensor, list, tuple)):
                x = args[0]
                is_array = True
            elif "x" in kwargs:
                x = kwargs["x"]
                is_array = True
            else:
                raise TypeError("Invalid arguments")

            if len(args) >= 2 and isinstance(args[1], (np.ndarray, torch.Tensor)):
                y = args[1]
                is_array = True
            elif "y" in kwargs:
                y = kwargs["y"]
                is_array = True
            else:
                raise TypeError("Invalid arguments")
        if is_array:
            train_data = (x, y)

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
                self._evaluate(validation_data=validation_data,
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

    def _checkpoint(self, save_best_only, restore_best_weights, epoch, epochs, best_epoch,
                    patience):
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
