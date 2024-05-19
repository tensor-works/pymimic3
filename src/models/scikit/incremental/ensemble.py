from tensorflow.keras.utils import Progbar
from models.scikit.incremental import AbstractIncrementalClassifier
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


if __name__ == '__main__':
    import datasets
    from tests.settings import *
    from preprocessing.imputers import PartialImputer
    from preprocessing.scalers import MIMICMinMaxScaler
    from generators.scikit import ScikitGenerator
    import torch
    from torch.utils.data import DataLoader

    class PyTorchDataLoaderStream:

        def __init__(self, dataloader: DataLoader):
            self.dataloader = dataloader
            self.iterator = iter(dataloader)
            self.n_samples = 0  # Total samples returned so far
            self.n_targets = 1  # Assuming binary classification

        def next_sample(self):
            try:
                batch = next(self.iterator)
            except StopIteration:
                # Reinitialize the iterator if it reaches the end
                self.iterator = iter(self.dataloader)
                batch = next(self.iterator)

            self.n_samples += 1
            X, y = batch
            return dict(enumerate(X.numpy().flatten())), y.numpy().item()

        def has_more_samples(self):
            return self.n_samples < len(self.dataloader.dataset)

        def n_remaining_samples(self):
            return len(self.dataloader.dataset) - self.n_samples

        def restart(self):
            self.iterator = iter(self.dataloader)
            self.n_samples = 0

        def get_data_info(self):
            return len(self.dataloader.dataset), self.n_targets

    # Example usage with PyTorch DataLoader
    from torch.utils.data import TensorDataset

    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                engineer=True,
                                task="IHM")
    imputer = PartialImputer().fit_reader(reader)
    scaler = MIMICMinMaxScaler(imputer=imputer).fit_reader(reader)
    generator = ScikitGenerator(reader=reader,
                                scaler=scaler,
                                batch_size=1,
                                drop_last=True,
                                shuffle=True)

    # Initialize the custom stream
    pytorch_stream = PyTorchDataLoaderStream(generator)

    # Example integration with river's AdaptiveRandomForestClassifier
    from river import ensemble, forest, metrics, preprocessing, stream

    # Initialize the classifier
    arf_clf = forest.ARFClassifier(seed=42)

    # Metric to evaluate the model
    from river import linear_model
    strd_clf = linear_model.LogisticRegression()
    import numpy as np
    metric = metrics.Accuracy()
    names = [str(i) for i in range(714)]
    for x, y in generator:
        x = np.squeeze(x)
        x = dict(zip(names, x))
        y = float(np.squeeze(y))
        y_pred = arf_clf.predict_one(x)
        metric.update(y, y_pred)
        arf_clf.learn_one(x, y)
    print(metric)

    for x, y in generator:
        x = np.squeeze(x)
        x = dict(zip(names, x))
        y = float(np.squeeze(y))
        y_pred = strd_clf.predict_one(x)
        metric.update(y, y_pred)
        strd_clf.learn_one(x, y)
    print(metric)
    """
    # Train and evaluate the model incrementally
    while pytorch_stream.has_more_samples():
        x, y = pytorch_stream.next_sample()
        y_pred = arf_clf.predict_one(x)
        metric = metric.update(y, y_pred)
        arf_clf = arf_clf.learn_one(x, y)
    
    """

    # Print the final metric
    print(f'Accuracy: {metric.get()}')
