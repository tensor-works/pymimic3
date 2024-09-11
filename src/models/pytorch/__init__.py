import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import warnings
from types import FunctionType
from typing import List, Tuple, Literal
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
        settings = {
            "task": self._task,
            "num_labels" if self._task == "multilabel" else "num_classes": self._num_classes
        }
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
        if ((len(args) >= 1 and isinstance(args[0], (np.ndarray, torch.Tensor, list, tuple))) or "x" in kwargs) \
            and (len(args) >= 2 and isinstance(args[1], (np.ndarray, torch.Tensor)) or "y" in kwargs):
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

        # Update metric history
        if hasattr(self._history, f"{prefix}_metrics"):
            metric_history = getattr(self._history, f"{prefix}_metrics")
            for key, metric in self._metrics[prefix].items():
                metric_history[key][epoch] = metric["value"]

        # Reset metric values
        for metric in self._metrics[prefix].values():
            metric["value"] = 0.0

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

        self._labels = list()
        self._outputs = list()
        '''

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
                finalize=finalize)  # and self._batch_count == self._epoch_progbar.target)

    def _prefixed_metrics(self, metrics: List[Tuple[str, float]],
                          prefix: str) -> List[Tuple[str, float]]:
        if prefix is None:
            return metrics
        return [(f"{prefix}_{metric}", value) for metric, value in metrics]

    def _remove_end_padding(self, tensor):
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
        # TODO! if the array is 2D does that mean B, T or B, N?
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
            output = self(input, masks=mask)

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
            output = self(input, masks=mask)

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

    def _optimize_batch(self, outputs: list, labels: list, finalize: bool = False):
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

    def _evaluate_with_arrays(self,
                              x: np.ndarray,
                              y: np.ndarray,
                              batch_size: int = None,
                              val_frequency: int = 1,
                              epoch: int = 0,
                              is_test: bool = False,
                              **kwargs):
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
        self._on_epoch_start(generator_size=len(y),
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
                    # TODO! Is there a better way. What if there is an actual patient with all zero across
                    # TODO! 59 columns? Am I paranoid. Maybe adding the discretizer masking can alleviat
                    val_inputs = self._remove_end_padding(val_inputs)
                    mask = None

                # TODO! add target replication here

                # Adjust dimensions
                val_inputs = val_inputs.unsqueeze(0)
                val_labels = val_labels.unsqueeze(0)

                # Create predictions
                val_outputs = self(val_inputs, masks=mask)

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
                        val_inputs, mask = val_inputs
                        val_inputs = val_inputs[mask.squeeze(), :]
                        val_inputs = val_inputs.to(self._device)
                        mask = mask.to(self._device)
                    else:
                        val_inputs = val_inputs.to(self._device)
                        mask = None
                    val_labels = val_labels.to(self._device)
                    val_outputs = self(val_inputs, masks=mask)
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

    def _evaluate_batch(self,
                        outputs: list,
                        labels: list,
                        is_test: bool = False,
                        finalize: bool = False):
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
            # TODO! update this logic
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

    @overload
    def evaluate(self, x: np.ndarray, y: np.ndarray):
        ...

    @overload
    def evaluate(self, validation_data: DataLoader):
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

    @overload
    def predict(self, x: np.ndarray, batch_size=None, verbose="auto", steps=None):
        ...

    @overload
    def predict(self, generator: DataLoader, batch_size=None, verbose="auto", steps=None):
        ...

    def predict(self, *args, batch_size=None, verbose="auto", steps=None, **kwargs):
        predict_data, is_array, args, kwargs = self._get_dataloader_or_array(*args,
                                                                             **kwargs,
                                                                             has_y=False)
        if is_array:
            return self._predict_numpy(*predict_data, batch_size=batch_size, verbose=verbose)
        return self._predict_dataloader(*predict_data, batch_size=batch_size, verbose=verbose)

    def _predict_dataloader(self,
                            generator: DataLoader,
                            batch_size=None,
                            verbose="auto",
                            steps=None,
                            **kwargs):
        self.eval()

        aggr_outputs = []

        with torch.no_grad():
            for idx, (inputs, label) in enumerate(generator):
                if masking_flag == None:
                    # Most efficient way to set this I could think of
                    masking_flag = isinstance(inputs, (list, tuple))

                if masking_flag:
                    inputs, mask = inputs
                    inputs = inputs.to(self._device)
                    # Set labels to zero when masking since forward does the same
                    mask = mask.to(self._device).bool()
                else:
                    inputs = inputs.to(self._device)
                    mask = None

                label = label.to(self._device).T
                output = self(inputs, masks=mask)

                # Accumulate outputs and labels
                aggr_outputs.append(output.to("cpu").numpy())
        return zeropad_samples(aggr_outputs)

    def _predict_numpy(self, x, batch_size=None, verbose="auto", steps=None, **kwargs):
        self.eval()
        # Handle masking
        if isinstance(x, (list, tuple)):
            x, masks = x
            masks = torch.tensor(masks, dtype=torch.float32).bool().to(self._device)
            masking_flag = True
        else:
            masking_flag = False

        # On device data
        x = torch.tensor(x, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            return self(x, masks=masks).to("cpu").detach().numpy()

    def _get_dataloader_or_array(self, *args, has_y=True, **kwargs):
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
