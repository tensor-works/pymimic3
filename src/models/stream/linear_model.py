from pathlib import Path
from models.stream import AbstractRiverModel, AbstractMultioutputClassifier
from river.linear_model import ALMAClassifier as _ALMAClassifier
from river.linear_model import LogisticRegression as _LogisticRegression
from river.linear_model import LinearRegression as _LinearRegression

__all__ = ["ALMAClassifier", "LogisticRegression"]


class ALMAClassifier(AbstractRiverModel, _ALMAClassifier):

    def __init__(self, model_path: Path = None, metrics: list = [], *args, **kwargs):
        self._default_name = "alma_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _ALMAClassifier.__init__(self, *args, **kwargs)


class LogisticRegression(AbstractRiverModel, _LogisticRegression):

    def __init__(self, model_path: Path = None, metrics: list = [], *args, **kwargs):
        self._default_name = "log_reg_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _LogisticRegression.__init__(self, *args, **kwargs)


class LinearRegression(AbstractRiverModel, _LinearRegression):

    def __init__(self, model_path: Path = None, metrics: list = [], *args, **kwargs):
        self._default_name = "log_reg_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        _LinearRegression.__init__(self, *args, **kwargs)


class MultiOutputLogisticRegression(AbstractRiverModel, AbstractMultioutputClassifier):

    def __init__(self, model_path: Path = None, metrics: list = [], *args, **kwargs):
        self._default_name = "log_reg_classifier"
        AbstractRiverModel.__init__(self, model_path, metrics)
        AbstractMultioutputClassifier.__init__(self, _LogisticRegression(*args, **kwargs))


if __name__ == '__main__':

    import datasets
    from river import optim
    from tests.tsettings import *
    from metrics.stream import MacroROCAUC, MicroROCAUC, PRAUC
    from river.metrics import ROCAUC
    from generators.stream import RiverGenerator
    from preprocessing.scalers import MinMaxScaler
    from preprocessing.imputers import PartialImputer

    ### Try LOS ###
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                engineer=True,
                                task="LOS")

    imputer = PartialImputer().fit_reader(reader)
    scaler = MinMaxScaler(imputer=imputer).fit_reader(reader)
    # reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)
    model = LinearRegression(metrics=["cohen_kappa", "mae"], optimizer=optim.SGD(0.00005))
    generator = RiverGenerator(reader, scaler=scaler)
    model.fit(generator)
    ### Try IHM ###
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                engineer=True,
                                task="IHM")

    imputer = PartialImputer().fit_reader(reader)
    scaler = MinMaxScaler(imputer=imputer).fit_reader(reader)
    # reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)
    model = LogisticRegression(metrics=[PRAUC, ROCAUC])
    generator = RiverGenerator(reader, scaler=scaler)
    model.fit(generator)

    ### Try PHENO ###
    reader = datasets.load_data(chunksize=75836,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                engineer=True,
                                task="PHENO")

    imputer = PartialImputer().fit_reader(reader)
    scaler = MinMaxScaler(imputer=imputer).fit_reader(reader)
    # reader = datasets.train_test_split(reader, test_size=0.2, val_size=0.1)
    model = MultiOutputLogisticRegression(metrics=[MacroROCAUC, MicroROCAUC])
    generator = RiverGenerator(reader, scaler=scaler)
    model.fit(generator)
    """
    metric1 = MacroROCAUC()
    metric2 = MicroROCAUC()
    for x, y in generator:
        model.learn_one(x, y)
        y_pred = model.predict_proba_one(x, predict_for=True)
        metric1.update(y, y_pred)
        metric2.update(y, y_pred)
        print(metric1, end="\r")
    print(metric1)
    print(metric2)
    """
