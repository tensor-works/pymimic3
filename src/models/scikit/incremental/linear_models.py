from tensorflow.keras.utils import Progbar
from . import AbstractIncrementalClassifier
from sklearn.linear_model import SGDClassifier
from utils import dict_subset


class IncrementalLogRegSKLearn(AbstractIncrementalClassifier):

    def __init__(self, task, alpha: float = 0.001, **kwargs) -> None:
        """_summary_

        Args:
            task (_type_): _description_
            alpha (float, optional): _description_. Defaults to 0.001.
        """
        super().__init__(task)
        self.clf = SGDClassifier(loss="log_loss", alpha=alpha, **kwargs)

    def _init_classifier(self, generator, classes):
        """_summary_

        Args:
            generator (_type_): _description_
            classes (_type_): _description_
        """
        X, y = generator.next()
        self.clf.partial_fit(X, y, classes=classes)

    def fit(self,
            generator,
            steps_per_epoch: int,
            epochs: int,
            validation_data=None,
            validation_steps: int = None):
        """_summary_

        Args:
            generator (_type_): _description_
            steps_per_epoch (_type_): _description_
            epochs (_type_): _description_
            validation_data (_type_): _description_
            validation_steps (_type_): _description_
        """
        tolerance = 1e-6
        early_stopping = 20

        classes_switch = {
            "DECOMP": [0, 1],
            "IHM": [0, 1],
            "LOS": [*range(10)],
            "PHENO": [*range(25)]
        }

        self._init_classifier(self, generator, classes=classes_switch[self.task])

        for _ in range(epochs):  # 10 passes through the data
            self._fit_iter(generator, steps_per_epoch)
            self._validation_epoch(validation_data, validation_steps)

    def compile(self, metrics, loss=None):
        """_summary_

        Args:
            metrics (_type_): _description_
            loss (_type_, optional): _description_. Defaults to None.
        """
        if loss is not None:
            self.loss = loss
        else:
            self.loss = self.model.loss_function_.py_loss

        self.iterative_metrics = dict_subset(self.iterative_metrics, metrics)
        self.iterative_metrics.update({"loss": self.loss})

    def _fit_iter(self, generator, steps_per_epoch):
        """_summary_

        Args:
            generator (_type_): _description_
            steps_per_epoch (_type_): _description_
        """
        progbar = Progbar(generator.steps,
                          unit_name='step',
                          stateful_metrics=[*self.iterative_metrics.keys()])

        for idx in range(steps_per_epoch):
            X, y_true = generator.next()
            self.clf.partial_fit(X, y_true)
            y_pred = self.clf.predict(X)
            metric_values = self._update_metrics(y_pred, y_true)
            progbar.update(idx, values=metric_values)

        self._reset_metrics()
