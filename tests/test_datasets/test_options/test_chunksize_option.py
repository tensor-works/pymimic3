import datasets
import os
import pandas as pd
import json
import shutil
import pytest
from itertools import chain
from typing import Dict
from pathlib import Path
from utils.IO import *
from tests.settings import *
from tests.utils.general import assert_dataframe_equals
from tests.utils.preprocessing import assert_reader_equals
from tests.utils.feature_engineering import extract_test_ids, concatenate_dataset

from datasets.readers import ExtractedSetReader, ProcessedSetReader
'''
def test_path_options():
    datasets.load_data(chunksize=5000000,
                       source_path=TEST_DATA_DEMO,
                       storage_path=TEMP_DIR,
                       preprocess=True,
                       task=generated_path.name)


def test_chunksize_option():
    ...
'''

if __name__ == "__main__":
    # test_num_samples_options()
    # extraction_reader = datasets.load_data(chunksize=75835,
    #                                        source_path=TEST_DATA_DEMO,
    #                                        preprocess=True,
    #                                        task="IHM",
    #                                        storage_path=SEMITEMP_DIR)

    # stest_num_subjects_extraction_only(extraction_reader)
    # test_num_subjects_preprocessing_only("IHM")
    if Path(TEMP_DIR, "engineered").is_dir():
        shutil.rmtree(str(Path(TEMP_DIR, "engineered")))
    test_num_subjects_engineer_only("IHM")
