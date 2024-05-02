import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils.IO import *
from utils import load_json, update_json
from tensorflow.keras import Model


class AbstractCheckpointManager(object):
    """_summary_
    """

    def __init__(self, directory):
        """_summary_
        """
        if isinstance(directory, str):
            self.directory = Path(directory)
        else:
            self.directory = directory

        self.custom_objects = []

    @property
    def latest(self):
        """_summary_
        """
        return self.latest_epoch()

    def load_model(self):
        """_summary_
        """
        model_path = Path(self.directory, f"cp-{self.latest:04d}.ckpt")
        info_io(f"Loading model from epoch {self.latest}.")
        model = tf.keras.models.load_model(model_path, self.custom_objects)

        return model

    def load_weights(self, model: Model):
        """_summary_

        Args:
            model (_type_): _description_
        """
        latest_cp_name = tf.train.latest_checkpoint(self.directory)
        model.load_weights(latest_cp_name)
        return model

    def latest_epoch(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError("This is an abstract class!")

    def clean_directory(self, best_epoch: int, keep_latest: bool = True):
        """_summary_

        Args:
            best_epoch (int): _description_
            keep_latest (bool, optional): _description_. Defaults to True.
        """
        raise NotImplementedError("This is an abstract class!")

    def is_empty(self):
        """_summary_
        """
        if self.latest == 0:
            return True
        return False


class CheckpointManager(AbstractCheckpointManager):
    """_summary_
    """

    def __init__(self, directory, train_epochs, custom_objects):
        """_summary_

        Args:
            directory (_type_): _description_
            train_epochs (_type_): _description_
            custom_objects (_type_): _description_
        """
        super().__init__(directory)
        self.epochs = train_epochs
        self.custom_objects = custom_objects

    def latest_epoch(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        check_point_epochs = [
            i for i in range(self.epochs + 1) for folder in self.directory.iterdir()
            if f"{i:04d}" in folder.name
        ]

        if check_point_epochs:
            return max(check_point_epochs)

        return 0

    def clean_directory(self, best_epoch: int, keep_latest: bool = True):
        """_summary_

        Args:
            best_epoch (int): _description_
            keep_latest (bool, optional): _description_. Defaults to True.
        """

        [
            shutil.rmtree(folder)
            for i in range(self.epochs + 1)
            for folder in self.directory.iterdir()
            if f"{i:04d}" in folder.name and ((i != self.epochs) or not keep_latest) and
            (i != best_epoch) and (".ckpt" in folder.name)
        ]


class ReducedCheckpointManager(AbstractCheckpointManager):

    def __init__(self, directory):
        """_summary_

        Args:
            directory (_type_): _description_
        """
        super().__init__(directory)

    def latest_epoch(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        items = [item for item in self.directory.iterdir()]
        check_point_epochs = [
            i for i in range(len(items)) for folder in items if f"{i:04d}" in folder.name
        ]

        if check_point_epochs:
            return max(check_point_epochs)

        return 0

    def clean_directory(self, best_epoch: int):
        """_summary_

        Args:
            best_epoch (int): _description_
            keep_latest (bool, optional): _description_. Defaults to True.
        """
        items = [item for item in self.directory.iterdir()]
        [
            shutil.rmtree(folder)
            for i in range(len(items))
            for folder in items
            if f"{i:04d}" in folder.name and (i != best_epoch) and (".ckpt" in folder.name)
        ]


class HistoryManager():
    """_summary_
    """

    def __init__(self, directory):
        """_summary_

        Args:
            directory (_type_): _description_
        """
        self.directory = directory
        self.history_file = Path(directory, "history.json")

    @property
    def history(self):
        self._history = load_json(self.history_file)
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    @property
    def best(self):
        """_summary_
        """
        if "val_loss" in self.history.keys():
            return min(self.history["val_loss"]), np.argmin(self.history["val_loss"]) + 1
        return None, None

    def update(self, items: dict):
        """_summary_v

        Args:
            items (dict): _description_
        """
        self._history = update_json(self.history_file, items)

    def is_finished(self):
        if "finished" in self.history.keys():
            return True
        return False

    def finished(self):
        """_summary_
        """
        self.update({'finished': True})
