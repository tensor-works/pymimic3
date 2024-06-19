"https://stackoverflow.com/questions/60727279/save-history-of-model-fit-for-different-epochs"
"https://stackoverflow.com/questions/69595923/how-to-decrease-the-learning-rate-every-10-epochs-by-a-factor-of-0-9"

import json
import tensorflow.keras.backend as K
from pathlib import Path
from tensorflow.keras import callbacks, backend


class HistoryCheckpoint(callbacks.Callback):
    """
    """

    def __init__(self, storage_path):
        """
        """
        self.storage_file = Path(storage_path, "history.json")
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """
        """

        if ('lr' not in logs.keys()):
            logs.setdefault('lr', 0)
            logs['lr'] = K.get_value(self.model.optimizer.lr)

        if self.storage_file.is_file():
            with open(self.storage_file, 'r+') as file:
                eval_hist = json.load(file)
        else:
            eval_hist = dict()

        for key, value in logs.items():
            if not key in eval_hist:
                eval_hist[key] = list()

            eval_hist[key].append(float(value))

        with open(self.storage_file, 'w') as file:
            json.dump(eval_hist, file, indent=4)

        return


class DecayLearningRate(callbacks.Callback):

    def __init__(self, freq, factor):
        """
        """
        self.freq = freq
        self.factor = factor

    def on_epoch_end(self, epoch, logs=None):
        """
        """
        if epoch % self.freq == 0 and not epoch == 0:  # adjust the learning rate
            lr = float(backend.get_value(self.model.optimizer.lr))  # get the current learning rate
            new_lr = lr * self.factor
            backend.set_value(self.model.optimizer.lr,
                              new_lr)  # set the learning rate in the optimizer
