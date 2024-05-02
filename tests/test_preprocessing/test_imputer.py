# Test the preprocessing.imputer class using the preprocessing_readers from conftest.py. You can find the imputer use case in preprocessing.discretizer and preprocessing.normalizer
import pytest
from preprocessing.imputers import BatchImputer
from tests.settings import *


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_imputer_fit_dataset(preprocessing_readers, task_name):
    # Arrange
    reader = preprocessing_readers()
    data = reader.get_random()  # assuming the reader has a read method that returns data
    imp = BatchImputer()  # replace with actual imputer initialization if different

    # Act
    imputed_data = imp.fit_transform(discretized_data)

    # Assert
    # replace with actual assertions based on your expectations
    assert imputed_data is not None
    assert imputed_data.isnull().sum().sum() == 0  # assuming imputer fills all NaN values


def test_imputer_with_normalizer():
    # Arrange
    reader = preprocessing_readers()
    data = reader.read()  # assuming the reader has a read method that returns data
    norm = normalizer.Normalizer()  # replace with actual normalizer initialization if different
    imp = imputer.Imputer()  # replace with actual imputer initialization if different

    # Act
    normalized_data = norm.fit_transform(data)
    imputed_data = imp.fit_transform(normalized_data)

    # Assert
    # replace with actual assertions based on your expectations
    assert imputed_data is not None
    assert imputed_data.isnull().sum().sum() == 0  # assuming imputer fills all NaN values
