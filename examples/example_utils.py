from utils import *
from datasets import ProcessedSetReader
from typing import Dict, List, Tuple
from examples.settings import YERVA_SPLIT
from datasets.readers import SplitSetReader


def print_split_info(reader: ProcessedSetReader, split_set: Dict[str, List[str]]):
    train_subjects = split_set["train"]
    val_subjects = split_set["val"]
    test_subjects = split_set["test"]
    train_percent = len(train_subjects) / len(reader.subject_ids) * 100
    val_percent = len(val_subjects) / len(reader.subject_ids) * 100
    test_percent = len(test_subjects) / len(reader.subject_ids) * 100

    max_len = max(len(train_subjects), len(val_subjects), len(test_subjects))
    width = len(str(max_len))

    info_io(f"Train: {len(train_subjects):<{width}} ({train_percent:0.2f}%) \n"
            f"Val:   {len(val_subjects):<{width}} ({val_percent:0.2f}%) \n"
            f"Test:  {len(test_subjects):<{width}} ({test_percent:0.2f}%)")


def benchmark_split_subjects() -> Tuple[List[int], List[int]]:
    print("Test and validation sets found...")
    test_subjects = pd.read_csv(Path(YERVA_SPLIT, "testset.csv"),
                                na_values=[''],
                                keep_default_na=False,
                                header=None)
    test_subjects.columns = ["subjects", "affiliation"]
    test_subjects = test_subjects[test_subjects["affiliation"] == 1]["subjects"].astype(
        int).tolist()

    val_subjects = pd.read_csv(Path(YERVA_SPLIT, "valset.csv"),
                               na_values=[''],
                               keep_default_na=False,
                               header=None)
    val_subjects.columns = ["subjects", "affiliation"]
    val_subjects = val_subjects[val_subjects["affiliation"] == 1]["subjects"].astype(int).tolist()
    return test_subjects, val_subjects


def benchmark_split_reader(reader: ProcessedSetReader, test_subjects: List[str],
                           val_subjects: List[str]) -> SplitSetReader:
    # Split data as in original set
    test_subjects = list(set(reader.subject_ids) & set(test_subjects))
    val_subjects = list(set(reader.subject_ids) & set(val_subjects))
    train_subjects = list(set(reader.subject_ids) - set(test_subjects) - set(val_subjects))

    split_sets = {"test": test_subjects, "val": val_subjects, "train": train_subjects}
    split_reader = SplitSetReader(reader.root_path, split_sets)

    # Print result
    print_split_info(reader, split_sets)
    return split_reader
