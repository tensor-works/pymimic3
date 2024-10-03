"""
Create all the datasets in the semi-temp directory for CI runs
"""

import pytest


@pytest.mark.usefixtures(
    "extracted_reader",
    "preprocessed_readers",
    "engineered_readers",
    "discretized_readers",
)
def test_all_fixtures():
    pass
