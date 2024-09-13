"""Utility file

This file implements functionalities used by other modules and accessible to 
the user

"""

import re
from utils.IO import *
from pathlib import Path


def to_snake_case(string):
    """
    Convert a string to snake case.

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    str
        The converted string in snake case.
    """
    # Replace spaces and hyphens with underscores
    string = re.sub(r'[\s-]+', '_', string)

    # Replace camel case and Pascal case with underscores
    string = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', string)

    # Convert the string to lowercase
    string = string.lower()

    return string


def count_csv_size(file_path: Path):

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        file_length = sum(bl.count("\n") for bl in blocks(f))

    return file_length - 1
