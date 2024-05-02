import datasets
import os
import pandas as pd
import json
import shutil
from pathlib import Path
from utils.IO import *
from tests.settings import *

base_path = Path(TEST_DATA_DIR, "configs", "case_configs")

test_config_paths = [
    Path(base_path, "logistic_decomp"),
    Path(base_path, "logistic_ihm"),
    Path(base_path, "logistic_los"),
    Path(base_path, "logistic_phenotyping"),
    Path(base_path, "logistic_subcases")
]

# def test_folder_creation_subcases():
