import os
import pandas as pd
from pathlib import Path

from utils.IO import *
from settings import *
from utils import make_prediction_vector
from preprocessing.preprocessors import MIMICPreprocessor
from visualization import make_history_plot


class AbstractMIMICPipeline(object):

    def __init__(self) -> None:
        ...

    def _processing(self, timeseries: pd.DataFrame, episodic_data: pd.DataFrame,
                    subject_diagnoses: pd.DataFrame, subject_icu_history: pd.DataFrame):
        """_summary_

        Args:
            timeseries (pd.DataFrame): _description_
            episodic_data (pd.DataFrame): _description_
            subject_diagnoses (pd.DataFrame): _description_
            subject_icu_history (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        info_io("Preprocessing data.")

        # TODO: outdated interface
        self.preprocessor = MIMICPreprocessor(config_dict=Path(os.getenv("CONFIG"),
                                                               "datasets.json"),
                                              task=self.task,
                                              storage_path=self.directories["task_data"],
                                              source_path=self.data_source_path,
                                              label_type="one-hot")

        (X_subjects,
         y_subjects) = self.preprocessor.transform(timeseries,
                                                   episodic_data,
                                                   subject_diagnoses,
                                                   subject_icu_history,
                                                   self.task,
                                                   task_path=self.directories["task_data"])
        if self.save:
            self.preprocessor.save_data()

        info_io("Done.")

        return X_subjects, y_subjects

    def _init_normalizer(self, train, preprocessor: str):
        """_summary_

        Args:
            X_train (_type_, optional): _description_. Defaults to None.
        """
        if not self.normalizer.load():
            if preprocessor == "discretizer":
                args = {"discretizer": self.discretizer}
            elif preprocessor == "imputer":
                args = {"imputer": self.imputer}
            info_io("Fitting normalizer.")
            if self.generator_flag:
                self.normalizer.fit_reader(train, **args)
            else:
                self.normalizer.fit_dataset(train, **args)
        else:
            info_io(f"Normalizer already fitted and stored at {self.normalizer.storage_path}")

        return

    def _init_imputer(self, train):
        """_summary_

        Args:
            X_train (_type_): _description_
        """
        if not self.imputer.load():
            info_io("Fitting imputer.")
            if self.generator_flag:
                self.imputer.fit_reader(train)
            else:
                self.imputer.fit(train)
        else:
            info_io(f"Imputer already fitted and stored at {self.imputer.storage_path}")

        return

    def _init_directories(self):
        """_summary_
        """
        is_subcase = False
        if not self.root_path:
            self.root_path = Path(os.getenv("MODEL"))

        if isinstance(self.root_path, str):
            self.root_path = Path(self.root_path)
            is_subcase = str(Path(self.root_path.parents[1], self.root_path.parents[1])) == str(
                Path(self.model_name, self.task))

        self.directories = {
            "root_path":
                self.root_path,
            "model_path":
                self.root_path if is_subcase else Path(self.root_path, self.model_name, self.task),
            "config_path":
                Path(os.getenv("CONFIG")),
            # ambiguous name but this is where loaded and processed data is stored
            #TODO! we assume, that the dataset only changes by the task, is why the normalizer shared
            #TODO! over all cases. This is a lil yanky
            "normalizer_path":
                Path(self.root_path, "normalizer", self.task),
            "imputer_path":
                Path(self.root_path, "imputer", self.task)
        }

        self.history_file = Path(self.directories["model_path"], "history.json")

        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return

    def _init_batch_reader_arguments(self):
        """_summary_
        """
        raise NotImplementedError("This method has not been implemented.")

    def _init_batch_readers(self, readers: object):
        """_summary_

        Args:
            task_reader (object): _description_
        """
        self._init_batch_reader_arguments()
        self.generators = {
            set_name: self._batch_reader(readers[set_name], **self.batch_reader_kwargs)
            for set_name in readers.keys()
        }

        return

    def _batch_reader(self, reader, **kwargs):
        """_summary_
        """
        raise NotImplementedError("This method has not been implemented.")

    def score(self, test_generator: object = None):
        """_summary_

        Args:
            test_generator (object, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if test_generator is None:
            test_generator = self.generators["test"]

        if test_generator is None:
            info_io("Pass a training fraction split precentage to score the model")
            return

        info_io("Evaluating model.")

        score = self.model.evaluate(test_generator, steps=test_generator.steps)
        self.hist_manager.update({"score": score})

        make_history_plot(self.hist_manager.history, self.directories["model_path"])

        info_io("Evaluation complete.")

        return score

    def compute_metrics(self, generator: object, metrics: dict, name: str = ""):
        """_summary_

        Args:
            generator (object): _description_
            metrics (dict): _description_
            name (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """
        y_pred, y_true = make_prediction_vector(self.model,
                                                generator,
                                                batches=generator.steps,
                                                bin_averages=self.preprocessor.bin_averages)

        return {(f"{name}_{key}" if name else key): metric(y_pred, y_true)
                for key, metric in metrics.items()}
