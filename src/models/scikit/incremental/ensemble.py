from tensorflow.keras.utils import Progbar
from . import AbstractIncrementalClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import dict_subset

# TODO!


class RandomForestClassifier(AbstractIncrementalClassifier):

    def __init__(self, task, n_estimators) -> None:
        super().__init__(task)
        self.clf = RandomForestClassifier(warm_start=True, n_estimators=n_estimators)
        self.n_estimators = n_estimators

    def _init_classifier(self, generator, classes):
        """_summary_

        Args:
            generator (_type_): _description_
            classes (_type_): _description_
        """
        pass

    def fit(self, generator, steps_per_epoch: int):
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

        self._fit_iter(generator, steps_per_epoch)

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
            self.clf.fit(X, y_true)
            self.clf.n_estimators += self.n_estimators
            y_pred = self.clf.predict(X)
            metric_values = self._update_metrics(y_pred, y_true)

            # values.append(("loss", clf.loss_function_.py_loss(y_pred, y)))
            progbar.update(idx, values=metric_values)

        self._reset_metrics()
