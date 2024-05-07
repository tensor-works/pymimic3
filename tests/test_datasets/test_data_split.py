import datasets
import random
import pandas as pd
from typing import List
from copy import deepcopy
from pathlib import Path
from utils.IO import *
from tests.settings import *

from datasets.readers import ProcessedSetReader
from datasets import train_test_split, SplitHandler, SplitUtility

####################################################################################################

import pandas as pd
import os
import yaml
import numpy as np
from sklearn import model_selection
from multipledispatch import dispatch
from collections import Iterable
from pathlib import Path
from datasets.trackers import DataSplitTracker, PreprocessingTracker
from utils import get_sample_size, dict_subset, load_json
from pathos.multiprocessing import Pool, cpu_count


class ReaderSplitter():

    def split_reader(self,
                     reader: ProcessedSetReader,
                     test_size: float = 0.0,
                     val_size: float = 0.0,
                     demographic_split: dict = None,
                     demographic_filter: dict = None):
        """_summary_

        Args:
            reader (ProcessedSetReader): _description_
            test_size (float): _description_
            val_size (float): _description_
        """
        info_io(f"Splitting reader", level=0)
        info_io(f"Target test size: {test_size}\nTarget validation size: {val_size}")
        info_io(f"Splitting in directory: {reader.root_path}")

        if demographic_filter is not None:
            subject_ids = self.get_demographics(reader.root_path, reader.subject_ids,
                                                demographic_filter)
        else:
            subject_ids = reader.subject_ids

        tracker = PreprocessingTracker(Path(reader.root_path, "progress"), subject_ids=subject_ids)
        self._tracker = DataSplitTracker(Path(reader.root_path, "split"), tracker, test_size,
                                         val_size)

        if demographic_split is not None and ("test" in demographic_split or "train"
                                              in demographic_split or "val" in demographic_split):
            subjects, ratios = self.split_by_demographics(reader, demographic_split)
            if test_size or val_size:
                self.reduce_by_ratio(subjects=subjects,
                                     ratios=ratios,
                                     test_size=test_size,
                                     val_size=val_size)
            for split in subjects:
                setattr(self._tracker, split, subjects[split])
            self._tracker.ratios = ratios
        else:
            subjects, ratios = self.split_by_ratio(test_size, val_size)
            for split in subjects:
                setattr(self._tracker, split, subjects[split])
            self._tracker.ratios = ratios

        message = ""
        for ratio in ratios:
            message += f"Real {ratio} size: {ratios[ratio]:0.3f}\n"
        info_io(message)

        self._tracker.is_finished = True

        return

    def reduce_by_ratio(self,
                        subjects: List[int],
                        ratios: List[int],
                        test_size: float = 0.0,
                        val_size: float = 0.0):
        self._tracker.ratios
        self._tracker.train
        self._tracker.val
        self._tracker.test

        return_subects = dict()
        return_ratios = dict()
        # if "test" in subjects and "val" in subjects:
        #     test_factor = test_size / ratios["test"]
        #     val_factor = val_size / ratios["val"]

        if "test" in subjects:
            if ratios["test"] < test_size:
                reduction_factor = ratios["test"] / test_size
                return_subects["train"] = random.sample(
                    subjects["train"], k=int(np.ceil(reduction_factor * len(subjects["train"]))))
            else:
                reduction_factor = test_size / ratios["test"]
                return_subects["test"] = random.sample(
                    subjects["test"],
                    k=int(np.ceil(0.5 / reduction_factor * len(subjects["test"]))))

        # reduction_factor =

    def split_by_demographics(self, reader: ProcessedSetReader, demographic_split: dict):
        return_ratios = dict()
        return_subjects = dict()
        for denominator, setting in demographic_split.items():
            assert denominator in ["test", "val", "train"], "Invalid split denominator"
            subject_ids = self.get_demographics(reader.root_path, reader.subject_ids, setting)
            return_ratios[denominator] = len(subject_ids) / len(reader.subject_ids)
            return_subjects[denominator] = subject_ids

        return return_subjects, return_ratios

    def split_by_ratio(self, test_size: float = 0.0, val_size: float = 0.0):
        return_ratios = dict()
        return_subjects = dict()
        subject_ratios = [(subject_id, self._tracker.subjects[subject_id]["total"])
                          for subject_id in self._tracker.subject_ids]
        ratio_df = pd.DataFrame(subject_ratios, columns=['participant', 'ratio'])
        total_len = ratio_df["ratio"].sum()
        ratio_df["ratio"] = ratio_df["ratio"] / total_len

        ratio_df = ratio_df.sort_values('ratio')

        def create_split(total_subjects, ratio_df, size):
            """_summary_
            """
            subjects, ratio = self.subjects_for_ratio(ratio_df, size)
            remaining_subjects = total_subjects - set(subjects)
            ratio_df = ratio_df[~ratio_df.index.isin(subjects)]
            return list(subjects), remaining_subjects, ratio_df, ratio

        if test_size:
            return_subjects["test"], \
            train_subjects, \
            ratio_df, \
            return_ratios["test"] = create_split(set(self._tracker.subject_ids), ratio_df,
                                                                   test_size)
        if val_size:
            val_size = val_size / (1 - test_size)
            return_subjects["val"],\
            train_subjects, \
            ratio_df, \
            return_ratios["val"] = create_split(
                train_subjects, ratio_df, val_size)
            return_ratios["val"] *= (1 - test_size)

        if train_subjects:
            return_subjects["test"] = list(train_subjects)[:]
            return_ratios["train"] = 1 - sum(list(return_ratios.values()))

        message = ""
        for ratio in return_ratios:
            message += f"Real {ratio} size: {return_ratios[ratio]:0.3f}\n"
        info_io(message)

        return

    def get_demographics(self, root_path: Path, subject_ids: list, settings: dict):
        subject_info_df = pd.read_csv(Path(root_path, "subject_info.csv"))
        subject_info_df = subject_info_df[subject_info_df["SUBJECT_ID"].isin(subject_ids)]

        for denominator, setting in settings.items():
            assert denominator in subject_info_df.columns, f"Invalid demographic. Choose from {*subject_info_df.columns,}"
            if "geq" in setting:
                if "leq" in setting:
                    assert not setting["geq"] > setting["leq"], "Invalid range"
                if "less" in setting:
                    assert not setting["geq"] > setting["less"], "Invalid range"
                assert not "greater" in setting, "Invalid setting, cannot have both less and leq"
            if "greater" in setting:
                if "leq" in setting:
                    assert not setting["greater"] > setting["leq"], "Invalid range"
                if "less" in setting:
                    assert not setting["greater"] > setting["less"], "Invalid range"
            if "leq" in setting and "less" in setting:
                raise ValueError("Invalid setting, cannot have both less and leq")
            if ("geq" in setting or "leq" in setting) and "choice" in setting:
                raise ValueError("Invalid setting, cannot have both range and choice")
            if "geq" in setting:
                subject_info_df = subject_info_df[subject_info_df[denominator] >= setting["geq"]]
            if "leq" in setting:
                subject_info_df = subject_info_df[subject_info_df[denominator] < setting["leq"]]
            if "choice" in setting:
                subject_info_df = subject_info_df[subject_info_df[denominator].isin(
                    setting["choise"])]
            if "greater" in setting:
                subject_info_df = subject_info_df[subject_info_df[denominator] > setting["greater"]]
            if "less" in setting:
                subject_info_df = subject_info_df[subject_info_df[denominator] < setting["less"]]
        return subject_info_df["SUBJECT_ID"].unique().tolist()

    def subjects_for_ratio(self, ratio_df: pd.DataFrame, target_size: float):
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

                current_size += sum(next_subject.ratio.iloc[0:sample_size])
                subject_name = next_subject.participant.iloc[0:sample_size]
                remaining_pairs_df = remaining_pairs_df.iloc[sample_size:]

                subjects.extend(subject_name)
                if remaining_pairs_df.empty:
                    break

                sample_size = int(
                    np.floor(abs(target_size_pr - current_size) / remaining_pairs_df.ratio.max()))

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
                                      list(range(max_iter)),
                                      chunksize=int(np.ceil(max_iter / n_cpus)))
            while best_diff > tolerance and iter < max_iter:
                diff, current_size, subjects = next(res)

                if diff < best_diff:
                    best_subjects, best_size, best_diff = subjects, current_size, diff

        return best_subjects, best_size


if __name__ == "__main__":
    reader = datasets.load_data(chunksize=75837,
                                source_path=TEST_DATA_DEMO,
                                storage_path=SEMITEMP_DIR,
                                discretize=True,
                                time_step_size=1.0,
                                start_at_zero=True,
                                impute_strategy='previous',
                                task='DECOMP')

    ret = datasets.train_test_split(  # reader,
        source_path=Path(SEMITEMP_DIR, "discretized", "DECOMP"),
        model_path=Path(SEMITEMP_DIR, "model"),
        test_size=0.2,
        split_info_path=Path(TEST_DATA_DEMO, "subject_info.csv"))

    # ReaderSplitter().split_reader(reader, test_size=0.2, val_size=0.2)

    # ReaderSplitter().split_reader(reader,
    #                               test_size=0.2,
    #                               val_size=0.2,
    #                               demographic_filter={"AGE": {
    #                                   "geq": 60
    #                               }})
    #
    # ReaderSplitter().split_reader(reader, test_size=0.2, demographic_filter={"AGE": {"geq": 60}})

    ReaderSplitter().split_reader(reader,
                                  test_size=0.2,
                                  demographic_split={
                                      "test": {
                                          "AGE": {
                                              "geq": 60
                                          }
                                      },
                                      "train": {
                                          "AGE": {
                                              "less": 60
                                          }
                                      }
                                  })

    pass
