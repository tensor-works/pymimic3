import random
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path
from utils.IO import *
from datasets.readers import ProcessedSetReader, SplitSetReader
from sklearn import model_selection
from pathlib import Path
from datasets.trackers import DataSplitTracker, PreprocessingTracker
from pathos.multiprocessing import Pool, cpu_count
from utils import dict_subset


class AbstractSplitter(object):

    def __init__(self, max_iter: int = 100, tolerance: float = 0.005) -> None:
        self._max_iter = max_iter
        self._tolerance = tolerance

    @staticmethod
    def _print_ratio(prefix: str, ratio: dict):
        message = [prefix + ":"]
        set_names = ["test", "val", "train"]
        for set_name in set_names:
            if set_name in ratio and ratio[set_name]:
                set_string = f" {set_name} size: ".ljust(12)
                message.append(f"{set_string}{ratio[set_name]:0.3f}")
        info_io("\n".join(message))

    def _reduce_by_ratio(self,
                         subjects: List[int],
                         ratios: List[int],
                         test_size: float = 0.0,
                         val_size: float = 0.0):
        if val_size and (not "val" in subjects or not subjects["val"]):
            raise ValueError(f"'val_size' parameter specified but no val subjects "
                             f"in reduce by ratio function! Val size is: {val_size}")

        set_ratios = dict()
        if test_size:
            set_ratios["test"] = test_size
        if val_size:
            set_ratios["val"] = val_size
        set_ratios["train"] = 1 - test_size - val_size

        sum_lenghts = sum([len(subjects[split]) for split in subjects])

        reduced = list()

        while True:
            reduction_factors = {
                set_name:
                set_ratios * sum([len(subjects[name]) for name in subjects if name != set_name]) /
                ((1 - set_ratios) * len(subjects[set_name]))
                for set_name, set_ratios in set_ratios.items()
                if not set_name in reduced
            }
            reduction_factors = {
                set_name: reduction_factor
                for set_name, reduction_factor in reduction_factors.items()
                if reduction_factor < 1
            }

            if not reduction_factors:
                break

            reduce_by = min(reduction_factors, key=lambda k: abs(reduction_factors[k] - 1))
            n_samples = int(np.ceil(reduction_factors[reduce_by] * len(subjects[reduce_by])))
            subjects[reduce_by] = random.sample(subjects[reduce_by], k=n_samples)

            sum_lenghts = sum([len(subjects[name]) for name in subjects])
            ratios = {set_name: len(subjects[set_name]) / sum_lenghts for set_name in subjects}
            reduced.append(reduce_by)

        return subjects, ratios

    def _split_by_demographics(self, subject_ids: List[int], source_path: Path,
                               demographic_split: dict):
        return_ratios = dict()
        return_subjects = dict()

        # Set wise demographics config for test, val and train
        for set_name, setting in demographic_split.items():
            assert set_name in ["test", "val", "train"], "Invalid split attribute"
            curr_subject_ids = self._get_demographics(set_name.capitalize(), source_path,
                                                      subject_ids, setting)
            return_ratios[set_name] = len(curr_subject_ids) / len(subject_ids)
            return_subjects[set_name] = curr_subject_ids

        # If not train specified select the substraction of the other sets
        if not "train" in demographic_split:
            train_subjects = set(subject_ids)
            prefix = "Train"
            for set_name, setting in demographic_split.items():
                train_subjects.intersection_update(
                    self._get_demographics(prefix, source_path, subject_ids, setting, invert=True))
                prefix = ""
            return_subjects["train"] = list(train_subjects)
            return_ratios["train"] = len(train_subjects) / len(subject_ids)

        return return_subjects, return_ratios

    def _split_by_ratio(self,
                        subject_ids: List[int],
                        subjects: dict,
                        test_size: float = 0.0,
                        val_size: float = 0.0):
        if test_size < 0 or test_size > 1:
            raise ValueError("Invalid test size")
        if val_size < 0 or val_size > 1:
            raise ValueError("Invalid val size")
        return_ratios = dict()
        return_subjects = dict()
        subject_ratios = [(subject_id, subjects[subject_id]["total"]) for subject_id in subject_ids]
        ratio_df = pd.DataFrame(subject_ratios, columns=['participant', 'ratio'])
        total_len = ratio_df["ratio"].sum()
        ratio_df["ratio"] = ratio_df["ratio"] / total_len

        ratio_df = ratio_df.sort_values('ratio')
        train_subjects = set(subject_ids)

        def create_split(total_subjects, ratio_df, size):
            """_summary_
            """
            subjects, split_ratio = self._subjects_for_ratio(ratio_df, size)
            remaining_subjects = total_subjects - set(subjects)
            updated_ratio_df = ratio_df[~ratio_df.participant.isin(subjects)]
            return list(subjects), remaining_subjects, updated_ratio_df, split_ratio

        if test_size:
            return_subjects["test"], \
            train_subjects, \
            ratio_df, \
            return_ratios["test"] = create_split(total_subjects=train_subjects,
                                                 ratio_df=ratio_df,
                                                 size=test_size)
        if val_size:
            return_subjects["val"],\
            train_subjects, \
            ratio_df, \
            return_ratios["val"] = create_split(total_subjects=train_subjects,
                                                ratio_df=ratio_df,
                                                size=val_size)

        if train_subjects:
            return_subjects["train"] = list(train_subjects)[:]
            return_ratios["train"] = 1 - sum(list(return_ratios.values()))

        return return_subjects, return_ratios

    def _get_demographics(self,
                          prefix: str,
                          source_path: Path,
                          subject_ids: list,
                          settings: dict,
                          invert=False):
        if source_path is None or settings is None:
            return subject_ids
        subject_info_df = pd.read_csv(Path(source_path, "subject_info.csv"))
        subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(subject_ids)]

        if prefix is not None and prefix:
            message = [prefix + ":"]
        else:
            message = []

        def get_subjects(condition: pd.Series):
            return subject_info_df[condition]["SUBJECT_ID"].unique()

        def get_categorical_message(choice):
            choice = list(choice)
            if len(choice) > 1:
                return f"is one of " + ", ".join(
                    str(entry) for entry in choice[:-1]) + " or " + str(choice[-1])
            else:
                return f"is {choice[0]}"

        def check_setting(setting, attribute):
            if not attribute in subject_info_df.columns:
                raise ValueError(f"Invalid demographic. Choose from {*subject_info_df.columns,}\n"
                                 f"Demographic is: {attribute}")
            if "geq" in setting:
                check_range(setting["geq"], setting)
                assert not "greater" in setting, "Invalid setting, cannot have both less and leq"

            if "greater" in setting:
                check_range(setting["greater"], setting)

            if "leq" in setting and "less" in setting:
                raise ValueError("Invalid setting, cannot have both less and leq")

            if ("geq" in setting or "leq" in setting or "less" in setting or
                    "greater" in setting) and "choice" in setting:
                raise ValueError("Invalid setting, cannot have both range and choice")

        def check_range(greater_value, setting):
            for key in ["leq", "less"]:
                if key in setting:
                    if greater_value > setting[key]:
                        raise ValueError(
                            f"Invalid range: greater={greater_value} > leq={setting[key]}")

        if invert:
            exclude_subjects = set(subject_ids)
        else:
            exclude_subjects = set()

        for attribute, setting in settings.items():
            attribute_message = " "
            check_setting(setting, attribute)
            attribute_data = subject_info_df[attribute]

            if "geq" in setting:
                # We use this reversed logic to avoid including any subjects where on stay
                # may fail the specification
                if invert:
                    exclude_subjects.intersection_update(
                        get_subjects(attribute_data >= setting["geq"]))
                    attribute_message += f"{setting['geq']:0.3f} > "
                else:
                    exclude_subjects.update(get_subjects(attribute_data < setting["geq"]))
                    attribute_message += f"{setting['geq']:0.3f} =< "

            if "greater" in setting:
                if invert:
                    exclude_subjects.intersection_update(
                        get_subjects(attribute_data > setting["greater"]))
                    attribute_message += f"{setting['greater']:0.3f} >= "
                else:
                    exclude_subjects.update(get_subjects(attribute_data <= setting["greater"]))
                    attribute_message += f"{setting['greater']:0.3f} < "

            if invert and ("geq" in setting or "greater" in setting) and\
                          ("leq" in setting or "less" in setting):
                attribute_message += f"{attribute} or "
            elif invert and not ("geq" in setting or "greater" in setting) and \
                                ("leq" in setting or "less" in setting):
                pass
            else:
                attribute_message += f"{attribute} "

            if "leq" in setting:
                if invert:
                    exclude_subjects.intersection_update(
                        get_subjects(attribute_data < setting["leq"]))
                    attribute_message += f"{setting['leq']:0.3f} < {attribute}"
                else:
                    exclude_subjects.update(get_subjects(attribute_data >= setting["leq"]))
                    attribute_message += f"<= {setting['leq']:0.3f}"

            if "less" in setting:
                if invert:
                    exclude_subjects.intersection_update(
                        get_subjects(attribute_data < setting["less"]))
                    attribute_message += f"{setting['less']:0.3f} <= {attribute}"
                else:
                    exclude_subjects.update(get_subjects(attribute_data >= setting["less"]))
                    attribute_message += f"< {setting['less']:0.3f}"

            if "choice" in setting:
                categories = attribute_data.unique()
                not_choices = set(categories) - set(setting["choice"])
                if invert:
                    exclude_subjects.intersection_update(
                        get_subjects(attribute_data.isin(setting["choice"])))
                    attribute_message += get_categorical_message(not_choices)
                else:
                    exclude_subjects.update(get_subjects(attribute_data.isin(not_choices)))
                    attribute_message += get_categorical_message(setting["choice"])

            message.append(attribute_message)

        if prefix is not None:
            info_io("\n".join(message))
        subject_info_df = subject_info_df[~subject_info_df["SUBJECT_ID"].isin(exclude_subjects)]
        return subject_info_df["SUBJECT_ID"].unique().tolist()

    def _subjects_for_ratio(self, ratio_df: pd.DataFrame, target_size: float):
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
        iter = 0

        def compute_ratios(random_state):
            current_size = 0
            remaining_pairs_df = ratio_df_pr
            subjects = list()
            sample_size = int(min(1, np.floor(target_size_pr / remaining_pairs_df.ratio.max())))

            while current_size < target_size_pr:
                current_to_rarget_diff = target_size_pr - current_size
                remaining_pairs_df = remaining_pairs_df[
                    remaining_pairs_df['ratio'] < current_to_rarget_diff]

                if remaining_pairs_df.empty:
                    break

                next_subject = remaining_pairs_df.sample(sample_size, random_state=random_state)

                current_size += sum(next_subject.ratio.tolist())
                subject_names = next_subject.participant.tolist()
                remaining_pairs_df = remaining_pairs_df[~remaining_pairs_df.participant.
                                                        isin(subject_names)]

                subjects.extend(subject_names)
                if remaining_pairs_df.empty:
                    break

                large_sample_bound = int(np.floor(
                                        abs(target_size_pr - current_size)\
                                        / remaining_pairs_df.ratio.max()))
                sample_size = min(large_sample_bound, len(remaining_pairs_df))

            diff = abs(target_size_pr - current_size)
            return diff, current_size, subjects

        def init(ratio_df, target_size):
            global ratio_df_pr, target_size_pr
            ratio_df_pr = ratio_df
            target_size_pr = target_size

        pool = Pool()  # Create a process pool
        n_cpus = cpu_count() - 1
        with Pool(n_cpus, initializer=init, initargs=(ratio_df, target_size)) as pool:
            res = pool.imap_unordered(compute_ratios,
                                      list(range(self._max_iter)),
                                      chunksize=int(np.ceil(self._max_iter / n_cpus)))
            while best_diff > self._tolerance and iter < self._max_iter:
                diff, current_size, subjects = next(res)
                iter += 1
                if diff < best_diff:
                    best_subjects, best_size, best_diff = subjects, current_size, diff

        return best_subjects, best_size


class ReaderSplitter(AbstractSplitter):

    def split_reader(self,
                     reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     demographic_split: dict = None,
                     demographic_filter: dict = None,
                     storage_path: Path = None):
        """_summary_

        Args:
            reader (ProcessedSetReader): _description_
            test_size (float): _description_
            val_size (float): _description_
        """
        info_io(f"Splitting reader", level=0)
        self._print_ratio("Target", {
            "train": 1 - test_size - val_size,
            "test": test_size,
            "val": val_size
        })
        info_io(f"Splitting in directory: {reader.root_path}")
        # Apply demographic filter
        subject_ids = self._get_demographics(prefix="Demographic filter",
                                             source_path=reader.root_path,
                                             subject_ids=reader.subject_ids,
                                             settings=demographic_filter)
        orig_subject_ids = set(subject_ids)
        # Get subject counts
        preprocessing_tracker = PreprocessingTracker(Path(reader.root_path, "progress"),
                                                     subject_ids=subject_ids)
        # Resotre split state
        storage_path = Path((storage_path \
                             if storage_path is not None \
                             else reader.root_path), "split")

        split_tracker = DataSplitTracker(storage_path,
                                         tracker=preprocessing_tracker,
                                         test_size=test_size,
                                         val_size=val_size,
                                         demographic_split=demographic_split,
                                         demographic_filter=demographic_filter)

        # Identical split settings
        if split_tracker.is_finished:
            return SplitSetReader(reader.root_path, split_tracker.split)

        # Split by demographics
        if demographic_split is not None and demographic_split:
            # Apply split
            split_dictionary, ratios = self._split_by_demographics(
                subject_ids=subject_ids,
                source_path=reader.root_path,
                demographic_split=demographic_split)
            # Enforce ratio
            if test_size or val_size:
                split_dictionary, ratios = self._reduce_by_ratio(subjects=split_dictionary,
                                                                 ratios=ratios,
                                                                 test_size=test_size,
                                                                 val_size=val_size)
            # Update tracker
            split_tracker.split = split_dictionary
            split_tracker.ratios = ratios
        else:
            # Split by ratio
            split_dictionary, ratios = self._split_by_ratio(subject_ids=subject_ids,
                                                            subjects=split_tracker.subjects,
                                                            test_size=test_size,
                                                            val_size=val_size)
            # Update tracker
            split_tracker.split = split_dictionary
            split_tracker.ratios = ratios

        self._print_ratio("Real", ratios)
        split_tracker.is_finished = True

        return SplitSetReader(reader.root_path, split_dictionary)


class CompactSplitter(AbstractSplitter):

    def split_dict(self,
                   X_subjects: Dict[str, Dict[str, pd.DataFrame]],
                   y_subjects: Dict[str, Dict[str, pd.DataFrame]],
                   test_size: float = 0.0,
                   val_size: float = 0.0,
                   demographic_split: dict = None,
                   demographic_filter: dict = None,
                   source_path: Path = None):
        """_summary_

        Args:
            reader (ProcessedSetReader): _description_
            test_size (float): _description_
            val_size (float): _description_
        """
        info_io(f"Splitting reader", level=0)

        target_ratios = {"train": 1 - test_size - val_size, "test": test_size, "val": val_size}
        message = ""
        for set_name in target_ratios:
            if target_ratios[set_name]:
                message += f"Target {set_name} size: {target_ratios[set_name]:0.3f}\n"
        info_io(message)
        info_io(f"Splitting in directory: {source_path}")
        if (demographic_filter is not None or demographic_split is not None) and not source_path:
            raise ValueError("Demographic split requires source path"
                             " to locate the subject_info.csv file")

        if demographic_filter is not None:
            subject_ids = self._get_demographics(source_path, list(X_subjects.keys()),
                                                 demographic_filter)
        else:
            subject_ids = list(X_subjects.keys())

        if source_path:
            tracker = PreprocessingTracker(Path(source_path, "progress"), subject_ids=subject_ids)
            split_tracker = DataSplitTracker(Path(source_path, "split"), tracker, test_size,
                                             val_size)
            if split_tracker.is_finished:
                ...
                # return SplitSetReader(source_path, )
        else:
            split_tracker = None

        if demographic_split is not None and ("test" in demographic_split or "train"
                                              in demographic_split or "val" in demographic_split):
            subjects, ratios = self._split_by_demographics(subject_ids, source_path,
                                                           demographic_split)
            if test_size or val_size:
                subjects, ratios = self._reduce_by_ratio(subjects=subjects,
                                                         ratios=ratios,
                                                         test_size=test_size,
                                                         val_size=val_size)
            if split_tracker is not None:
                for split in subjects:
                    setattr(split_tracker, split, subjects[split])
                split_tracker.ratios = ratios
        else:
            subject_counts = self._compute_subject_counts(y_subjects)
            subjects, ratios = self._split_by_ratio(subject_ids, subject_counts, test_size,
                                                    val_size)
            if split_tracker is not None:
                for split in subjects:
                    setattr(split_tracker, split, subjects[split])
                split_tracker.ratios = ratios

        message = ""
        for set_name in ratios:
            message += f"Real {set_name} size: {ratios[set_name]:0.3f}\n"
        info_io(message)

        if split_tracker is not None:
            split_tracker.is_finished = True
        dataset = {
            set_name: (dict_subset(X_subjects, subjects), dict_subset(y_subjects, subjects))
            for set_name, subjects in subjects.items()
        }

        return dataset

    def _compute_subject_counts(self, y_subjects: Dict[str, Dict[str, pd.DataFrame]]):
        subject_counts = {
            subject_id: {stay_id: len(stay_data) for stay_id, stay_data in subject_data.items()
                        } for subject_id, subject_data in y_subjects.items()
        }
        for subject in subject_counts:
            subject_counts[subject]["total"] = sum(subject_counts[subject].values())
        return subject_counts
