"""Dataset file

This file allows access to the dataset as specified.
All function in this file are used by the main interface function load_data.
Subfunctions used within private functions are located in the datasets.utils module.

Todo:
    - Use a settings.json
    - This is a construction site, see what you can bring in here
    - Provid link to kaggle in load_data doc string
    - Expand function to utils

YerevaNN/mimic3-benchmarks
"""
import pandas as pd
import os
import yaml
import numpy as np
from sklearn import model_selection
from multipledispatch import dispatch
from collections import Iterable
from pathlib import Path
from trackers import DataSplitTracker
from utils import get_sample_size, dict_subset, load_json
from . import extraction
from . import preprocessing
from . import feature_engineering
from .readers import ProcessedSetReader, ExtractedSetReader
from settings import *
from utils.IO import *

# global settings

__all__ = ["load_data", "train_test_split"]


def load_data(source_path: str,
              storage_path: str = None,
              chunksize: int = None,
              subject_ids: list = None,
              num_subjects: int = None,
              num_samples: int = None,
              extract: bool = True,
              preprocess: bool = False,
              engineer: bool = False,
              task: str = None):
    """_summary_

    Args:
        stoarge_path (str, optional): Location where the processed dataset is to be stored. Defaults to None.
        source_path (str, optional): Location form which the unprocessed dataset is to be loaded. Defaults to None.
        ehr (str, optional): _description_. Defaults to None.
        from_storage (bool, optional): _description_. Defaults to True.
        chunksize (int, optional): _description_. Defaults to None.
        num_subjects (int, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    storage_path = Path(storage_path)
    source_path = Path(source_path)

    subject_ids = _check_inputs(storage_path=storage_path,
                                source_path=source_path,
                                chunksize=chunksize,
                                subject_ids=subject_ids,
                                num_subjects=num_subjects,
                                num_samples=num_samples,
                                extract=extract,
                                preprocess=preprocess,
                                engineer=engineer,
                                task=task)

    # Iterative generation if a chunk size is specified
    if storage_path is not None and chunksize is not None:
        if extract or preprocess or engineer:
            extracted_storage_path = Path(storage_path, "extracted")
            # Account for missing subjects
            reader = extraction.iterative_extraction(storage_path=extracted_storage_path,
                                                     source_path=source_path,
                                                     chunksize=chunksize,
                                                     subject_ids=subject_ids,
                                                     num_subjects=num_subjects,
                                                     num_samples=num_samples,
                                                     task=task)

        if preprocess or engineer:
            # Contains phenotypes and a list of codes referring to the phenotype
            with Path(source_path, "resources", "hcup_ccs_2015_definitions.yaml").open("r") as file:
                phenotypes_yaml = yaml.full_load(file)

            processed_storage_path = Path(storage_path, "processed", task)
            reader = preprocessing.iterative_processing(reader=reader,
                                                        task=task,
                                                        subject_ids=subject_ids,
                                                        num_subjects=num_subjects,
                                                        num_samples=num_samples,
                                                        storage_path=processed_storage_path,
                                                        phenotypes_yaml=phenotypes_yaml)

        if engineer:
            engineered_storage_path = Path(storage_path, "engineered", task)
            reader = feature_engineering.iterative_fengineering(
                subject_ids=subject_ids,
                num_subjects=num_subjects,
                num_samples=num_samples,
                reader=reader,
                task=task,
                storage_path=engineered_storage_path)

        return reader

    elif chunksize is not None:
        raise ValueError("To run iterative iteration, specify storage path!")

    # Compact generation otherwise
    if extract or preprocess or engineer:
        extracted_storage_path = Path(storage_path, "extracted")
        dataset = extraction.compact_extraction(storage_path=extracted_storage_path,
                                                source_path=source_path,
                                                num_subjects=num_subjects,
                                                num_samples=num_samples,
                                                subject_ids=subject_ids,
                                                task=task)
    if preprocess or engineer:
        processed_storage_path = Path(storage_path, "processed", task)
        # Contains phenotypes and a list of codes referring to the phenotype
        with Path(source_path, "resources", "hcup_ccs_2015_definitions.yaml").open("r") as file:
            phenotypes_yaml = yaml.full_load(file)
        dataset = preprocessing.compact_processing(dataset=dataset,
                                                   task=task,
                                                   subject_ids=subject_ids,
                                                   num_subjects=num_subjects,
                                                   num_samples=num_samples,
                                                   storage_path=processed_storage_path,
                                                   phenotypes_yaml=phenotypes_yaml)

    if engineer:
        engineered_storage_path = Path(storage_path, "engineered", task)
        dataset = feature_engineering.compact_fengineering(dataset["X"],
                                                           dataset["y"],
                                                           task=task,
                                                           storage_path=engineered_storage_path,
                                                           subject_ids=subject_ids,
                                                           num_subjects=num_subjects)

    # TODO: make dependent from return reader (can also return reader)
    # TODO: write some tests for comparct generation
    return dataset


def _check_inputs(storage_path: str, source_path: str, chunksize: int, subject_ids: list,
                  num_subjects: int, num_samples: int, extract: bool, preprocess: bool,
                  engineer: bool, task: str):
    if chunksize and not storage_path:
        raise ValueError(f"Specify storage path if using iterative processing!"
                         f"Storage path is '{storage_path}' and chunksize is '{chunksize}'")
    if (preprocess or engineer) and not task:
        raise ValueError(
            "Specify the 'task' parameter for which to preprocess or engineer the data!"
            " Possible values for task are: DECOMP, LOS, IHM, PHENO")
    if task and not (engineer or preprocess):
        warn_io(f"Specified  task '{task}' for data extraction only, despite "
                "data extraction being task agnostic. Parameter is ignored.")
    if subject_ids and num_subjects:
        raise ValueError("Specify either subject_ids or num_subjects, not both!")
    if num_samples and (subject_ids or num_subjects):
        raise ValueError("Specify either num_sample_sets or subject_ids/num_subjects, not both!")
    if not any([extract, preprocess, engineer]):
        raise ValueError("One of extract, preprocess or engineer must be set to load the dataset.")
    if subject_ids is not None:
        return [int(subject_id) for subject_id in subject_ids]
    return None


def train_test_split(X=None,
                     y=None,
                     test_fraction_split: float = 0.,
                     validation_fraction_split: float = 0.,
                     dates: list = [],
                     mapping: bool = False,
                     split_info_path: pd.DataFrame = None,
                     progress_file_path: dict = None,
                     subgroup: str = None,
                     splitgroup: str = None,
                     concatenate: bool = True,
                     source_path: Path = None,
                     model_path: Path = None):
    """_summary_

    Args:
        X (_type_, optional): _description_. Defaults to None.
        y (_type_, optional): _description_. Defaults to None.
        test_size (float, optional): _description_. Defaults to 0.5.
        val_size (float, optional): _description_. Defaults to 0..
        dates (list, optional): _description_. Defaults to [].
        method (str, optional): _description_. Defaults to "sample".
        concatenate (bool, optional): _description_. Defaults to True.
        source_path (Path, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    su = SplitUtility()

    return su.train_test_split(X=X,
                               y=y,
                               test_size=test_fraction_split,
                               val_size=validation_fraction_split,
                               dates=dates,
                               mapping=mapping,
                               subgroup=subgroup,
                               splitgroup=splitgroup,
                               split_info_path=split_info_path,
                               progress_file_path=progress_file_path,
                               concatenate=concatenate,
                               source_path=source_path,
                               model_path=model_path)


def is_split(path: Path):
    """_summary_

    Args:
        path (Path): _description_
    """
    split_file = Path(path, "split.json")
    if split_file.is_file():
        if load_json(split_file)["finished"]:
            return True
    return False


class SplitUtility():

    def __init__(self):
        """
        """
        ...

    def train_test_split(self, X, y, dates: list, split_info_path: pd.DataFrame,
                         progress_file_path: Path, subgroup: str, splitgroup: str, mapping: bool,
                         concatenate: bool, source_path: Path, model_path: Path, test_size: float,
                         val_size: float):
        """This function splits the provided data into a test and train data set.

        Args:
            X (_type_): _description_
            y (_type_): _description_
            test_size (float): _description_
            val_size (float): _description_
            dates (list): _description_
            method (str): _description_
            concatenate (bool): _description_
            source_path (_type_, optional): _description_. Defaults to Path.

        Returns:
            _type_: _description_
        """

        def reader_dictionary(subject_sets):
            return {
                set_name: ProcessedSetReader(source_path, subject_folders=subjects)
                for set_name, subjects in subject_sets.items()
            }

        if splitgroup:
            handler = SplitHandler(split_info_path, progress_file_path)
            subject_sets, ratios = handler.split(subgroup, train=splitgroup)

            return reader_dictionary(subject_sets), ratios

        if source_path is not None:
            subject_sets, ratios = self.split_folders(source_path, model_path, test_size, val_size)

            return reader_dictionary(subject_sets), ratios

        test_size = float(test_size)
        val_size = float(val_size)

        if isinstance(X, dict) and len(X) == 1:
            if y is not None:
                y = [*y.values()][0]
                return self.make_frame_split([*X.values()][0],
                                             y,
                                             test_size=test_size,
                                             val_size=val_size)
            else:
                return self.make_frame_split([*X.values()][0],
                                             test_size=test_size,
                                             val_size=val_size)

        elif isinstance(X, dict):
            if not y:
                y = {}
            return self.make_dictionary_split(X, y, test_size, val_size)

        elif isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            if y is not None:
                return self.make_frame_split(X, y, test_size=test_size, val_size=val_size)
            else:
                return self.make_frame_split(X, test_size=test_size, val_size=val_size)

    @dispatch(Iterable)
    def make_frame_split(self, X: pd.DataFrame, test_size: float, val_size: float):
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            test_size (float): _description_
            val_size (float): _description_
        """
        total_size = test_size + val_size
        X_dataset = dict()
        X_dataset["train"], X_dataset["test"] = model_selection.train_test_split(
            X, test_size=total_size, shuffle=False)

        if val_size:
            val_test_ratio = val_size / total_size

            X_dataset["test"], X_dataset["val"] = model_selection.train_test_split(
                X_dataset["test"], test_size=val_test_ratio, shuffle=False)

        return X_dataset, None

    @dispatch(Iterable, Iterable)
    def make_frame_split(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float, val_size: float):
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            y (pd.DataFrame): _description_
            test_size (float): _description_
            val_size (float): _description_
        """
        total_size = test_size + val_size
        X_dataset = dict()
        y_dataset = dict()
        X_dataset["train"], \
        X_dataset["test"], \
        y_dataset["train"], \
        y_dataset["test"] = model_selection.train_test_split(X,
                                                                y,
                                                                test_size=total_size,
                                                                shuffle=False)

        if val_size:
            val_test_ratio = val_size / total_size

            X_dataset["test"], \
            X_dataset["val"], \
            y_dataset["test"], \
            y_dataset["val"] = model_selection.train_test_split(X_dataset["test"],
                                                                   y_dataset["test"],
                                                                   test_size=val_test_ratio,
                                                                   shuffle=False)

        return X_dataset, y_dataset

    def unpack_subjects(self, subject_dict):
        return_data = list()
        [
            return_data.append(subject_data) if not isinstance(subject_data, dict) else
            return_data.extend(self.unpack_subjects(subject_data))
            for subject_data in subject_dict.values()
        ]

        return return_data

    def make_dictionary_split(self,
                              X_dict: dict,
                              y_dict: dict,
                              test_size: float = 0.5,
                              val_size: float = 0,
                              concatenate: bool = True):
        """_summary_

        Args:
            X_dict (dict): _description_
            y_dict (dict): _description_
            test_size (float, optional): _description_. Defaults to 0.5.
            val_size (float, optional): _description_. Defaults to 0.
            concatenate (bool, optional): _description_. Defaults to True.
        """
        subject_splits = dict()
        X_dataset = dict()
        y_dataset = dict()
        subject_splits["test"], _ = self.get_subset_subjects(X_dict, test_size)

        if test_size > 0 and not len(subject_splits["test"]):
            raise ("Test set empty!")

        subject_splits["train"] = set(X_dict.keys()) - set(subject_splits["test"])

        if val_size:
            X_train_dict = {key: X_dict[key] for key in subject_splits["train"]}
            val_train_ratio = val_size / (1 - test_size)
            subject_splits["val"], _ = self.get_subset_subjects(X_train_dict, val_train_ratio)

            if val_size > 0 and not len(subject_splits["val"]):
                raise ("Test set empty!")

            subject_splits["train"] = subject_splits["train"] - set(subject_splits["val"])

        for set_name, subjects in subject_splits.items():
            X_dataset[set_name] = self.unpack_subjects(dict_subset(X_dict, subjects))
            if len(y_dict):
                y_dataset[set_name] = self.unpack_subjects(dict_subset(y_dict, subject_splits))

        return X_dataset, y_dataset

    def get_subset_subjects(self, X_dict: dict, target_size: float):
        """_summary_

        Args:
            X_dict (dict): _description_
            target_size (float): _description_

        Returns:
            _type_: _description_
        """
        total_samples = get_sample_size(X_dict)

        # Create a dataframe containing the participant id and share on total sample as ratio
        # TODO! unteseted logic to detect nesting
        # TODO! consider recursion
        subject_ratios = [
            (subject, get_sample_size(data) / total_samples) for subject, data in X_dict.items()
        ]

        # Build dataframe
        ratio_df = pd.DataFrame(subject_ratios, columns=['participant', 'ratio'])
        ratio_df = ratio_df.sort_values('ratio')

        return self.deduce_subjects_byratio(ratio_df, target_size)

    def deduce_subjects_byratio(self, ratio_df: pd.DataFrame, target_size: float):
        """_summary_

        Args:
            ratio_df (pd.DataFrame): _description_
            target_size (float): _description_

        Returns:
            _type_: _description_
        """
        assert "participant" in ratio_df.columns
        assert "ratio" in ratio_df.columns
        best_diff = 1e18
        tolerance = 0.005
        max_iter = 1000
        iter = 0
        random_state = 0

        while best_diff > tolerance and iter < max_iter:

            current_size = 0
            remaining_pairs_df = ratio_df
            random_state += 1
            subjects = list()

            while current_size < target_size:
                current_to_rarget_diff = target_size - current_size
                remaining_pairs_df = remaining_pairs_df[
                    remaining_pairs_df['ratio'] < current_to_rarget_diff]

                if not len(remaining_pairs_df):
                    break

                next_subject = remaining_pairs_df.sample(1, random_state=random_state)

                current_size += next_subject.ratio.iloc[0]
                subject_name = next_subject.participant.iloc[0]
                remaining_pairs_df = remaining_pairs_df[
                    remaining_pairs_df.participant != subject_name]

                subjects.append(subject_name)

            diff = abs(target_size - current_size)

            if diff < best_diff:
                best_subjects, best_size, best_diff = subjects, current_size, diff

            iter += 1

        return best_subjects, best_size

    def split_folders(self,
                      source_path: Path,
                      model_path,
                      test_size: float = 0.5,
                      val_size: float = 0.):
        """_summary_

        Args:
            source_path (Path): _description_
            test_size (float, optional): _description_. Defaults to 0.5.
            val_size (float, optional): _description_. Defaults to 0..
        """
        tracker = DataSplitTracker(source_path, model_path, test_size, val_size)
        return_ratios = dict()
        info_io(f"Splitting dataset")
        info_io(f"Storing split info at {str(Path(model_path, 'split.json'))}")
        if tracker.finished:
            subject_sets = {
                set_name: list(map(lambda x: Path(source_path, x), subjects))
                for set_name, subjects in zip(["train", "val", "test"],
                                              [tracker.train, tracker.validation, tracker.test])
                if subjects is not None
            }
            info_io("Dataset is already split")
            set_size_msg = " - ".join(
                ["Subjects per set"] +
                [f"{key}: {len(value)}" for key, value in subject_sets.items()])
            info_io(set_size_msg)
            ratio_size_msg = " - ".join(
                ["Real split ratios"] +
                [f"{key}: {value:.4f}" for key, value in tracker.ratios.items()])
            info_io(ratio_size_msg)
            return subject_sets, tracker.ratios

        train_subjects = tracker.subjects & tracker.directories

        n_missing_subjects = len(tracker.subjects) - len(tracker.directories)
        if n_missing_subjects:
            info_io(f"Missing subject directories: {n_missing_subjects}")

        subject_ratios = [
            (subject_id, tracker.subject_data[subject_id]["total"]) for subject_id in train_subjects
        ]
        ratio_df = pd.DataFrame(subject_ratios, columns=['participant', 'ratio'])
        total_len = ratio_df["ratio"].sum()
        ratio_df["ratio"] = ratio_df["ratio"] / total_len

        ratio_df = ratio_df.sort_values('ratio')

        def create_split(name, total_subjects, ratio_df, size):
            """_summary_
            """

            folder = Path(source_path, name)
            subjects, ratio = self.deduce_subjects_byratio(ratio_df, size)

            remaining_subjects = total_subjects - set(subjects)
            ratio_df = ratio_df[~ratio_df.index.isin(subjects)]
            return list(subjects), remaining_subjects, ratio_df, ratio

        dataset = dict()

        if test_size:
            test_subjects, \
            train_subjects, \
            ratio_df, \
            return_ratios["test"] = create_split("test", train_subjects, ratio_df,
                                                                   test_size)
            tracker.test = test_subjects
            dataset["test"] = list(map(lambda x: Path(source_path, x), test_subjects))
            info_io(f"Test set with {len(dataset['test'])} subjects")

        if val_size:
            val_size = val_size / (1 - test_size)
            validation_subjects,\
            train_subjects, \
            ratio_df, \
            return_ratios["val"] = create_split(
                "val", train_subjects, ratio_df, val_size)
            return_ratios["val"] *= (1 - test_size)
            tracker.validation = validation_subjects
            dataset["val"] = list(map(lambda x: Path(source_path, x), validation_subjects))
            info_io(f"Validation set with {len(dataset['val'])} subjects")

        if train_subjects:
            train_subjects = list(train_subjects)[:]
            tracker.train = train_subjects
            dataset["train"] = list(map(lambda x: Path(source_path, x), train_subjects))
            return_ratios["train"] = 1 - sum(list(return_ratios.values()))
            info_io(f"Train set with {len(dataset['train'])} subjects")

        tracker.ratios = return_ratios
        tracker.finished = True

        return dataset, return_ratios


class SplitHandler(object):

    def __init__(self, split_info_path, progress_file_path):
        self._split_info_df = pd.read_csv(Path(split_info_path))
        self._subject_sample_counts = load_json(progress_file_path)["subjects"]
        self._count_dfs = dict()
        self._others = dict()
        self._collapse_mappings = dict()
        self._unique_info_df = self._split_info_df.drop_duplicates(subset='SUBJECT_ID',
                                                                   keep="first")

        self._colum_settings = {
            "AGE": {
                "binning": 5,
                "count_na": False
            },
            "ETHNICITY": {
                "collapse": ["WHITE", "BLACK", "HISPANIC", "ASIAN"],
                "other": 500
            },
            "GENDER": {},
            "LANGUAGE": {
                "other": 50
            },
            "INSURANCE": {},
            "DISCHARGE_LOCATION": {},
            "ADMISSION_LOCATION": {},
            "WARDID": {},
            "CAREUNIT": {},
            "DBSOURCE": {}
        }
        self._process_attributes()

    def _value_counts(self, column: str, cout_na: bool):
        # TODO! keeps only first subject
        counts_df = self._unique_info_df.groupby([column]).size()
        if cout_na:
            counts_df["NAN"] = self._unique_info_df[column].isna().astype(int).sum()
        counts_df.name = "size"
        return counts_df.reset_index()

    def _bin_counts(self, column: str, counts_df: pd.DataFrame, bin_size: int):
        bins = np.linspace(self._split_info_df[column].min(), 90, int(90 / bin_size) + 1)
        groups = counts_df.groupby(pd.cut(counts_df[column], bins))["size"]
        return groups.sum().reset_index()

    def _collapse_counts(self, column: str, counts_df: pd.DataFrame, collapse_columns: list):
        collapse_mapping = dict()
        for collapse_column in collapse_columns:
            index = counts_df[column].apply(lambda x: collapse_column in x)
            collapse_mapping[collapse_column] = counts_df[index][column].tolist()
            total_count = counts_df[index]["size"].sum()
            counts_df = counts_df[~index].reset_index(drop=True)

            counts_df.loc[len(counts_df)] = [collapse_column, total_count]

        return counts_df.sort_values("size", ascending=False), collapse_mapping

    def _other_counts(self, column: str, counts_df: pd.DataFrame, min_counts: int):
        others = counts_df[(counts_df["size"] <= min_counts) |
                           (counts_df[column].str.upper() == "OTHER") |
                           (counts_df[column].str.upper() == "OTHERS")]
        total_count = others["size"].sum()
        counts_df = counts_df[(counts_df["size"] > min_counts)
                              & (counts_df[column].str.upper() != "OTHER")
                              & (counts_df[column].str.upper() != "OTHERS")]

        counts_df.loc[len(counts_df)] = ["OTHER", total_count]
        return counts_df.sort_values("size", ascending=False), others[column].to_list()

    def _process_attributes(self):
        for attribute, setting in self._colum_settings.items():
            count_na = "count_na" not in setting
            counts_df = self._value_counts(attribute, count_na)
            if "binning" in setting:
                counts_df = self._bin_counts(attribute, counts_df, setting["binning"])
            if "collapse" in setting:
                counts_df, self._collapse_mappings[attribute] = self._collapse_counts(
                    attribute, counts_df, setting["collapse"])
            if "other" in setting:
                counts_df, self._others[attribute] = self._other_counts(
                    attribute, counts_df, setting["other"])
            self._count_dfs[attribute] = counts_df

    def _compute_ratio(self, subjects_set: dict, subset_subjects: list = None):
        if subset_subjects:
            subset_counts = np.array(subset_counts, dtype=str).tolist()
            subset_counts = dict_subset(self._subject_sample_counts, subjects_set)
            total_count = np.array(
                [subject_count["total"] for subject_count in subset_counts.values()]).sum()
        else:
            total_count = np.array([
                subject_count["total"] for subject_count in self._subject_sample_counts.values()
            ]).sum()

        def count_samples(subjects: list):
            subjects = np.array(subjects, dtype=str).tolist()
            sample_counts = dict_subset(self._subject_sample_counts, subjects)
            sample_count = np.array(
                [subject_count["total"] for subject_count in sample_counts.values()]).sum()
            return sample_count / total_count

        return {set_name: count_samples(subjects) for set_name, subjects in subjects_set.items()}

    def get_subjects(self, attribute, items=None):
        if isinstance(items, Iterable) and not isinstance(items, str):
            items = list(items)
            if attribute in self._others and "OTHER" in items:
                del items[items.index("OTHER")]
                items.extend(self._others[attribute])
            if attribute in self._collapse_mappings:
                for key, value in self._collapse_mappings[attribute].items():
                    if not key in items:
                        continue
                    del items[items.index(key)]
                    items.extend(value)
            return self._unique_info_df[self._unique_info_df[attribute].isin(
                items)]["SUBJECT_ID"].unique().tolist()
        elif isinstance(items, str):
            if attribute in self._others and "OTHER" == items:
                items = self._others[attribute]
            elif attribute in self._collapse_mappings and items in self._collapse_mappings[
                    attribute]:
                items = self._collapse_mappings[attribute][items]
            if not isinstance(items, list):
                items = [items]
            return self._unique_info_df[self._unique_info_df[attribute].isin(
                items)]["SUBJECT_ID"].unique().tolist()
        elif items is None:
            return self._unique_info_df["SUBJECT_ID"].unique().tolist()
        else:
            raise ValueError(f"Passed {type(items)} to split handler.")

    def split(self, attribute, test=None, val=None, train=None, on_subset=None):
        subject_sets = dict()
        eval_set = list()
        if on_subset is not None:
            on_subset = self.get_subjects(attribute, on_subset)
        if test is not None:
            subject_sets["test"] = self.get_subjects(attribute, test)
            eval_set.extend(subject_sets["test"])
        if val is not None:
            subject_sets["val"] = self.get_subjects(attribute, val)
            eval_set.extend(subject_sets["val"])
        if train is not None:
            subject_sets["train"] = self.get_subjects(attribute, train)
        else:
            if on_subset is not None:
                subject_sets["train"] = list(set(on_subset) - set(eval_set))
            else:
                subject_sets["train"] = list(set(self.get_subjects(attribute)) - set(eval_set))

        if test is None and val is None and train is not None:
            if on_subset is not None:
                subject_sets["test"] = list(set(on_subset) - set(subject_sets["train"]))
            else:
                subject_sets["test"] = list(
                    set(self.get_subjects(attribute)) - set(subject_sets["train"]))

        return subject_sets, self._compute_ratio(subject_sets, on_subset)


if __name__ == "__main__":
    resource_folder = Path(os.getenv("WORKINGDIR"), "datalab", "mimic", "data_splits", "resources")
    handler = SplitHandler(Path(resource_folder, "subject_info_df.csv"),
                           Path(resource_folder, "progress.json"))
    assert len(handler.get_subjects("ETHNICITY", "WHITE")) == 30019
    assert len(handler.get_subjects("ETHNICITY", "BLACK")) == 3631
    assert len(handler.get_subjects("ETHNICITY", "UNKNOWN/NOT SPECIFIED")) == 3861
    assert len(handler.get_subjects("ETHNICITY", "HISPANIC")) == 1538
    assert len(handler.get_subjects("ETHNICITY", "ASIAN")) == 1623
    assert len(handler.get_subjects("ETHNICITY", "OTHER")) == 1902
    assert len(handler.get_subjects("ETHNICITY", "UNABLE TO OBTAIN")) == 703

    assert len(handler.get_subjects("ETHNICITY", ["WHITE", "BLACK"])) == 30019 + 3631
    assert len(handler.get_subjects("ETHNICITY", ["HISPANIC", "ASIAN"])) == 1538 + 1623

    dataset, ratios = handler.split("ETHNICITY", test="WHITE", train="BLACK")
    assert len(dataset["train"]) == 3631
    assert len(dataset["test"]) == 30019

    dataset, ratios = handler.split("ETHNICITY", test="WHITE")
    assert sum(list(ratios.values())) == 1
    assert len(dataset["train"]) == 3631 + 3861 + 1538 + 1623 + 1902 + 703
    assert len(dataset["test"]) == 30019

    dataset, ratios = handler.split("ETHNICITY", train="WHITE")
    assert sum(list(ratios.values())) == 1
    assert len(dataset["train"]) == 30019
    assert len(dataset["test"]) == 3631 + 3861 + 1538 + 1623 + 1902 + 703

    dataset, ratios = handler.split("LANGUAGE", train=["ENGL", "SPAN"])
    assert sum(list(ratios.values())) == 1
    assert len(dataset["train"]) == 19436 + 728
