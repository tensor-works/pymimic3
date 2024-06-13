import pandas as pd
import numpy as np
from utils.IO import *
from utils import is_numerical, is_colwise_numerical


def assert_dataframe_equals(generated_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            rename: dict = {},
                            normalize_by: str = "generated",
                            compare_mode: str = "single_entry"):
    """Compares the dataframes and print out the results. The age column cannot be compared, as the results are
    erroneous in the original dataset.

    Args:
        generated_df (pd.DataFrame): DataFrame generated by the MIMIC-III-Rework
        test_df (pd.DataFrame): DataFrame generated by the original MIMIC-III code.
        rename (dict, optional): Renaming of columns if necessary. Defaults to {}.
        normalize_by (str, optional): Define by which dataframe to structure the comparision. Can be 'generated' or 'groundtruth'. Defaults to "generated".
        compare_mode (str, optional): Define wether to compare data in absolute or proxemity terms. Data from numpy data files should be compared using the 
                                      proxemity mode. Can be 'absolute' or 'proxemity'. Defaults to "abolute".
    """
    assert normalize_by in ['generated', 'groundtruth'
                           ], "normalize_by needs to be one of 'generated' or 'groundtruth'."
    assert compare_mode in ['single_entry', 'multiline'
                           ], "compare_mode needs to be one of 'single_entry' or 'multiline'."

    # Assert shape 2
    assert len(generated_df.columns) == len(test_df.columns), (
        f"Generated and ground truth dataframes do not have the same amount of columns.\n"
        f"Generated: {len(generated_df.columns)}, Ground truth: {len(test_df.columns)}.")

    # Assert shape 1
    if not len(generated_df) == len(test_df):
        tests_io("Dataframe has diff!")
        difference = pd.concat([test_df, generated_df]).drop_duplicates(keep=False)
        tests_io(
            f"Length generated: {len(generated_df)}\nLength test: {len(test_df)}\nDiff is: \n{difference}"
        )
        assert len(generated_df) == len(test_df), (
            f"Generated and ground truth dataframes do not have the same amount of rows.\n"
            f"Generated: {len(generated_df)}, Ground truth: {len(test_df)}.")

    generated_df = generated_df.rename(columns=rename)
    if normalize_by == "generated":
        # Age erroneous in test code
        if "AGE" in generated_df:
            generated_df = generated_df.drop(columns=["AGE"])
        elif "Age" in test_df:
            generated_df = generated_df.drop(columns=["Age"])
        test_df = test_df[generated_df.columns]
    else:
        # Age erroneous in test code
        if "Age" in test_df:
            test_df = test_df.drop(columns=["Age"])
        elif "AGE" in test_df:
            test_df = test_df.drop(columns=["AGE"])
        generated_df = generated_df[test_df.columns]

    # Check dtypes TODO! used to work but fails on extraction because IDs are floats
    # assert (test_df.dtypes == generated_df.dtypes).all(), (
    #     f"Generated and ground truth dataframes do not have the same dtypes.\n"
    #     f"Generated: {generated_df.dtypes[(test_df.dtypes != generated_df.dtypes)]}, Ground truth: { test_df.dtypes[(test_df.dtypes != generated_df.dtypes)]}."
    # )

    # Check performed column wise
    if compare_mode == "single_entry":
        difference = 0
        for col in generated_df.select_dtypes(include=['category']).columns:
            if not "NoValue" in generated_df[col].cat.categories:
                generated_df[col] = generated_df[col].cat.add_categories('NoValue')
        for col in test_df.select_dtypes(include=['category']).columns:
            if not "NoValue" in test_df[col].cat.categories:
                test_df[col] = test_df[col].cat.add_categories('NoValue')
        frame_diff = (generated_df.fillna("NoValue") != test_df.fillna("NoValue"))
        for column_name in frame_diff:
            if not frame_diff[column_name].any():
                continue
            gen_col = generated_df[[column_name]]
            test_col = test_df[[column_name]]

            # Absolute vs threshold comparision depending on dtype:
            if is_numerical(gen_col) or is_numerical(test_col):
                # Checking for diffs with general float type
                # Casting float because pandas float and integer are not supported
                # Pandas has no in-house implementation of isclose
                column_diff = np.squeeze(
                    ~np.isclose(gen_col.astype(float), test_col.astype(float), equal_nan=True))
            else:
                column_diff = np.squeeze((gen_col.fillna("NoValue") != \
                                          test_col.fillna("NoValue")).values)
            if column_diff.shape == ():
                column_diff = column_diff.reshape(1)
            # If all identical continue
            if not column_diff.any():
                continue
            difference += 1
            display_diff_df = pd.concat([gen_col, test_col], axis=1)
            display_diff_df.index = gen_col.index
            display_diff_df.columns = ["Generated", "Groundtruth"]
            display_diff_df = display_diff_df[column_diff]
            tests_io(f"For column {column_name}:\n"
                     f"-------------------\n"
                     f"{display_diff_df}")
    elif compare_mode == "multiline":
        # For every subject and stay id
        stay_count = 0
        difference = 0
        sample_count = 0
        subject_buffer = list()
        if "SUBJECT_ID" in generated_df and "ICUSTAY_ID" in generated_df:
            is_numerical_dict = is_colwise_numerical(test_df)
            is_numerical_dict = {
                column: is_numerical_dict[column] or value
                for column, value in is_colwise_numerical(generated_df).items()
            }
            tests_io(f"Total stays checked: {stay_count}\n"
                     f"Total subjects checked: {len(set(subject_buffer))}\n"
                     f"Total samples checked: {sample_count}\n"
                     f"Total differences found: {difference}")
            for subject_id in generated_df["SUBJECT_ID"].unique():
                # Get all rows with the same subject_id
                subject_data = generated_df[generated_df["SUBJECT_ID"] == subject_id]
                test_subject_data = test_df[test_df["SUBJECT_ID"] == subject_id]
                for icustay_id in subject_data["ICUSTAY_ID"].unique():
                    # Get all rows with the same icustay_id
                    gen_rows = subject_data[subject_data["ICUSTAY_ID"] == icustay_id]
                    test_rows = test_subject_data[test_subject_data["ICUSTAY_ID"] == icustay_id]
                    for _, gen_row in gen_rows.iterrows():
                        diff = compare_homogenous_multiline_df(gen_row.to_frame().T, test_rows)

                        if diff:
                            tests_io(
                                f"For file {subject_id}_episode{icustay_id}_timeseries.csv, no equivalent entry in test_df found:\n"
                                f"-------------------\n"
                                f"{gen_row}")
                            difference += 1
                        sample_count += 1
                        tests_io(
                            f"Total stays checked: {stay_count}\n"
                            f"Total subjects checked: {len(set(subject_buffer))}\n"
                            f"Total samples checked: {sample_count}\n"
                            f"Total differences found: {difference}",
                            flush_block=True)

                    # Progress variables
                    stay_count += 1
                    subject_buffer.append(subject_id)
        else:
            # Find a matching row in test_df with the same index
            difference = compare_homogenous_multiline_df(generated_df, test_df, verbose=True)

    assert not difference, f"Diffs detected between generated and ground truth files: {difference}!"


def split_dataframes_by_type(df: pd.DataFrame, numerical_dict: dict):
    numerical_cols = [col for col, is_num in numerical_dict.items() if is_num]
    non_numerical_cols = [col for col, is_num in numerical_dict.items() if not is_num]
    return df[numerical_cols].astype(float), df[non_numerical_cols].astype("object")


def compare_homogenous_multiline_df(generated_df: pd.DataFrame,
                                    test_df: pd.DataFrame,
                                    verbose: bool = False):
    numerical_dict = is_colwise_numerical(test_df)
    numerical_dict.update({
        column: numerical_dict.get(column, False) or value
        for column, value in is_colwise_numerical(generated_df).items()
    })

    # Split both DataFrames
    gen_num_df, gen_non_num_df = split_dataframes_by_type(generated_df, numerical_dict)
    test_num_df, test_non_num_df = split_dataframes_by_type(test_df, numerical_dict)

    difference = 0
    sample_count = 0

    # Iterate over generated DataFrame rows
    for idx in gen_num_df.index:
        gen_num_row = gen_num_df.loc[idx]
        gen_non_num_row = gen_non_num_df.loc[idx]
        num_match = np.isclose(test_num_df.astype(float).values,
                               gen_num_row.astype(float).values,
                               equal_nan=True).all(axis=1)
        non_num_match = (
            gen_non_num_row.fillna("NoValue") == test_non_num_df.fillna("NoValue")).all(
                axis=1).values
        found = (num_match & non_num_match).any()
        if not found:
            # I don't remember why I did this, its been over a month
            difference += 1
            raise LookupError(
                f"No equivalent entry found for row index {idx}:\n{generated_df.loc[idx].to_frame().T}"
            )
        sample_count += 1
        if verbose:
            info_io(f"Total samples checked: {sample_count}\nTotal differences found: {difference}",
                    flush_block=True)

    return difference
