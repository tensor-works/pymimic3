import pytest
import pandas as pd
import numpy as np
from tests.utils.general import assert_dataframe_equals
from copy import deepcopy


def generate_random_frame(n_rows, n_cols, numeric=False):
    # Generate random data of different types for the DataFrame

    data = {
        f"col_{i}": np.where(
            np.random.rand(n_rows) < 0.2,
            np.nan,  # Inject NaNs
            np.random.randint(0, 100, size=n_rows) if i % 3 == 0 else np.random.rand(n_rows) *
            100 if i % 3 == 1 or numeric else np.random.choice(['A', 'B', 'C', 'D'], size=n_rows))
        for i in range(n_cols)
    }

    # Create the DataFrame
    df_random = pd.DataFrame(data)
    return df_random


def generate_random_series(dtypes: pd.Series):
    series_data = pd.Series(index=dtypes.index, dtype=object)

    for col, dtype in dtypes.items():
        dtype_name = dtype.name
        if dtype_name.startswith('int'):
            series_data[col] = np.random.randint(0, 100)
        elif dtype_name.startswith('float'):
            series_data[col] = np.random.random() * 100
        elif dtype_name == 'object':
            series_data[col] = ''.join(
                np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) for _ in range(5))
        elif dtype_name == 'datetime64[ns]':
            series_data[col] = pd.Timestamp('2020-01-01') + pd.to_timedelta(
                np.random.randint(0, 365), unit='D')
        elif dtype_name == 'category':
            categories = ['A', 'B', 'C', 'D']
            series_data[col] = pd.Categorical([np.random.choice(categories)], categories=categories)
        else:
            series_data[col] = np.nan

    return series_data


def test_compare_dataframes():
    nested_mock_columns = {
        "int": {
            "1": [1, 2, 3],
            "2": [1, 4, 3]
        },
        "float": {
            "1": [1.0, 2.5, 3.5],
            "2": [1.0, 2.4, 3.5]
        },
        "string1": {
            "1": ["a", "b", "c"],
            "2": ["a", "B", "c"]
        },
        "string2": {
            "1": ["a", "b", "c"],
            "2": ["a", "bb", "c"]
        },
        "np_int": {
            "1": np.array([1, 2, 3]),
            "2": np.array([1, 4, 3])
        },
        "np_float": {
            "1": np.array([1.0, 2.5, 3.5]),
            "2": np.array([1.0, 2.4, 3.5])
        },
        "pandas_categorical": {
            "1": pd.Categorical(["test", "train", "test"], categories=["test", "train", "val"]),
            "2": pd.Categorical(["test", "train", "val"], categories=["test", "train", "val"])
        }
    }

    # Test single entry differences
    base_frame_df = generate_random_frame(3, 3, True)
    assert_dataframe_equals(base_frame_df, deepcopy(base_frame_df))
    for noramlized_by in ["generated", "groundtruth"]:
        for dtype, columns in nested_mock_columns.items():
            type_1_df = deepcopy(base_frame_df)
            type_1_df["col_3"] = columns["1"]
            type_2_df = deepcopy(base_frame_df)
            type_2_df["col_3"] = columns["2"]
            assert_dataframe_equals(type_1_df,
                                    type_1_df,
                                    normalize_by=noramlized_by,
                                    compare_mode="single_entry")
            assert_dataframe_equals(type_2_df,
                                    type_2_df,
                                    normalize_by=noramlized_by,
                                    compare_mode="single_entry")
            with pytest.raises(AssertionError) as error:
                assert_dataframe_equals(type_1_df,
                                        type_2_df,
                                        normalize_by=noramlized_by,
                                        compare_mode="single_entry")
                assert str(
                    error.value) == "Diffs detected between generated and ground truth files: 1!"

            with pytest.raises(AssertionError) as error:
                assert_dataframe_equals(type_2_df,
                                        type_1_df,
                                        normalize_by=noramlized_by,
                                        compare_mode="single_entry")
                assert str(
                    error.value) == "Diffs detected between generated and ground truth files: 1!"

        # Test single entry dimension mismatch
        long_frame = deepcopy(base_frame_df)
        long_frame.loc[len(long_frame)] = generate_random_series(long_frame.dtypes)
        assert long_frame.shape == (4, 3)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    long_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="single_entry")
            assert str(error.value) == ("Generated and ground truth dataframes do not have"
                                        " the same amount of rows. Generated: 3, Ground truth: 4.")

        short_frame = deepcopy(base_frame_df)
        short_frame = short_frame.drop(short_frame.index[-1])
        assert short_frame.shape == (2, 3)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    short_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="single_entry")
            assert str(error.value) == ("Generated and ground truth dataframes do not have"
                                        " the same amount of rows. Generated: 3, Ground truth: 2.")

        wide_frame = deepcopy(base_frame_df)
        wide_frame["col_3"] = generate_random_series(pd.Series([long_frame.dtypes[0]] * 3))
        assert wide_frame.shape == (3, 4)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    wide_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="single_entry")
            assert str(error.value) == ("Generated and ground truth dataframes do not have the "
                                        "same amount of columns. Generated: 3, Ground truth: 4.")

        narrow_frame = deepcopy(base_frame_df)
        narrow_frame = narrow_frame.iloc[:, :-1]
        assert narrow_frame.shape == (3, 2)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    narrow_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="single_entry")
            assert str(error.value) == ("Generated and ground truth dataframes do not have the "
                                        "same amount of columns. Generated: 3, Ground truth: 2.")

    # Test multi entry differences
    base_frame_df = pd.concat([generate_random_frame(3, 3) for _ in range(3)])
    base_frame_df.index = [f"{idx}_episode{idx}_timeseries.csv" for idx in base_frame_df.index]
    assert_dataframe_equals(base_frame_df, deepcopy(base_frame_df))
    for noramlized_by in ["generated", "groundtruth"]:
        for _, columns in nested_mock_columns.items():
            if "string" in dtype or "categorical" in dtype:
                # This is done for numeric dfs from the feature engine so no objects
                continue
            type_1_df = deepcopy(base_frame_df)
            type_1_df["col_3"] = columns["1"] * 3
            type_2_df = deepcopy(base_frame_df)
            type_2_df["col_3"] = columns["2"] * 3
            assert_dataframe_equals(type_1_df,
                                    type_1_df,
                                    normalize_by=noramlized_by,
                                    compare_mode="multiline")
            assert_dataframe_equals(type_2_df,
                                    type_2_df,
                                    normalize_by=noramlized_by,
                                    compare_mode="multiline")
            with pytest.raises(AssertionError) as error:
                assert_dataframe_equals(type_1_df,
                                        type_2_df,
                                        normalize_by=noramlized_by,
                                        compare_mode="multiline")
                assert str(
                    error.value) == "Diffs detected between generated and ground truth files: 1!"

            with pytest.raises(AssertionError) as error:
                assert_dataframe_equals(type_2_df,
                                        type_1_df,
                                        normalize_by=noramlized_by,
                                        compare_mode="multiline")
                assert str(
                    error.value) == "Diffs detected between generated and ground truth files: 1!"

        # Test single entry dimension mismatch
        long_frame = deepcopy(base_frame_df)
        long_frame.loc[len(long_frame)] = generate_random_series(long_frame.dtypes)
        assert long_frame.shape == (10, 3)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    long_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="multiline")
            assert str(error.value) == ("Generated and ground truth dataframes do not have"
                                        " the same amount of rows. Generated: 9, Ground truth: 10.")

        short_frame = deepcopy(base_frame_df)
        short_frame = short_frame.iloc[0:len(short_frame) - 1]
        assert short_frame.shape == (8, 3)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    short_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="multiline")
            assert str(error.value) == ("Generated and ground truth dataframes do not have"
                                        " the same amount of rows. Generated: 9, Ground truth: 8.")

        wide_frame = deepcopy(base_frame_df)
        wide_frame["col_3"] = generate_random_series(pd.Series([long_frame.dtypes[0]] * 3))
        assert wide_frame.shape == (9, 4)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    wide_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="multiline")
            assert str(error.value) == ("Generated and ground truth dataframes do not have the "
                                        "same amount of columns. Generated: 9, Ground truth: 4.")

        narrow_frame = deepcopy(base_frame_df)
        narrow_frame = narrow_frame.iloc[:, :-1]
        assert narrow_frame.shape == (9, 2)
        with pytest.raises(AssertionError) as error:
            assert_dataframe_equals(base_frame_df,
                                    narrow_frame,
                                    normalize_by=noramlized_by,
                                    compare_mode="multiline")
            assert str(error.value) == ("Generated and ground truth dataframes do not have the "
                                        "same amount of columns. Generated: 9, Ground truth: 2.")


if __name__ == "__main__":
    test_compare_dataframes()
