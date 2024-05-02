import numpy as np
from tensorflow.keras.metrics import AUC, Accuracy
from copy import deepcopy


class AbstractIncrementalClassifier(object):

    def __init__(self, task) -> None:
        """_summary_

        Args:
            clf (_type_): _description_
            task (_type_): _description_
        """
        self.task = task
        self.iterative_metrics = {
            "accuracy": Accuracy(),
            "auc_macro": AUC(curve="ROC"),
            "auc_micro": AUC(curve="PR"),
        }
        self.best_score = np.infty
        self.best_model = None
        self.counter = 0

    def _init_classifier(self, generator, classes):
        raise NotImplementedError("This is an abstract method.")

    def _fit_iter(self, generator, steps_per_epoch):
        """_summary_

        Args:
            generator (_type_): _description_
            steps_per_epoch (_type_): _description_
        """
        raise NotImplementedError("This is an abstract method.")

    def _reset_metrics(self):
        """_summary_
        """
        [metric.reset_state() for metric in self.iterative_metrics.values()]

    def _update_metrics(self, y_pred, y_true):
        """_summary_

        Args:
            y_pred (_type_): _description_
            y_true (_type_): _description_
        """
        [metric.update_state(y_pred, y_true) for metric in self.iterative_metrics.values()]
        return [(metric_name, metric.result().numpy())
                for metric_name, metric in self.iterative_metrics.items()]

    def _validation_epoch(self, validation_data, validation_steps):
        """_summary_

        Args:
            validation_data (_type_): _description_
            validation_steps (_type_): _description_
        """
        for idx in range(validation_steps):
            X, y_true = validation_data.next()
            y_pred = self.clf.predict(X)
            self._update_metrics(y_true, y_pred)

        print("".join([
            f" - val_{metric_name}: {metric.result().numpy():.4f}"
            for metric_name, metric in self.iterative_metrics.items()
        ]))

        self._reset_metrics()

    def _early_stopping(self, score_value, patience):
        """_summary_

        Args:
            score_value (_type_): _description_
            patience (_type_): _description_

        Returns:
            _type_: _description_
        """
        if score_value > self.best_score:
            self.best_score = score_value
            self.best_model = deepcopy(self.clf)
            self.counter = 0
        elif score_value > self.best_score - score_value:
            self.counter = 0
        elif self.counter >= patience:
            return True
        else:
            self.counter += 1
        return False
