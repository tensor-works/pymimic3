"""
Collection of utility functions for the MIMIC-III dataset extraction process.

YerevaNN/mimic3-benchmarks

Functions
---------
- copy_subject_info(source_path, storage_path)
    Copy subject information from source path to storage path.
- get_samples_per_df(event_frames, num_samples)
    Get the number of samples per DataFrame based on the specified number of samples.
- convert_dtype_value(value, dtype)
    Convert a value to the specified dtype.
- convert_dtype_dict(dtypes, add_lower=True)
    Convert a dictionary of column names to dtype strings to a dictionary of column names to dtype objects.
- clean_chartevents_util(chartevents)
    Clean the chartevents DataFrame based on the timeseries column.
- get_static_value(timeseries, variable)
    Get the first non-null value of a specified variable from the time series data.
- upper_case_column_names(frame)
    Convert the column names to upper case for consistency.
- convert_to_numpy_types(frame)
    Convert the dtypes to numpy types for consistency.
- read_varmap_csv(resource_folder)
    Read the variable map CSV file from the specified resource folder.
- _clean_height(df)
    Convert height from inches to centimeters.
- _clean_systolic_bp(df)
    Filter out systolic blood pressure only.
- _clean_diastolic_bp(df)
    Filter out diastolic blood pressure only.
- _clean_capilary_rr(df)
    Categorize capillary refill rate.
- _clean_fraction_inspired_o2(df)
    Map fraction of inspired oxygen values to correct scale.
- _clean_laboratory_values(df)
    Clean laboratory values by removing non-numeric entries.
- _clean_o2sat(df)
    Scale oxygen saturation values to correct range.
- _clean_temperature(df)
    Map Fahrenheit temperatures to Celsius.
- _clean_weight(df)
    Convert weight values to kilograms.
- _clean_respiratory_rate(df)
    Transform respiratory rate values from greater than 60 to 60.
"""

import numpy as np
import pandas as pd
import os
import re
import shutil
from pathlib import Path
from typing import Dict
from utils.IO import *
from settings import *


def copy_subject_info(source_path: Path, storage_path: Path):
    """
    Copy subject information from source path to storage path.

    Parameters
    ----------
    source_path : Path
        The path to the source directory containing the subject information file.
    storage_path : Path
        The path to the target directory where the subject information file will be copied.
    """
    if source_path is None:
        if storage_path is not None:
            warn_io("No source path provided for subject information. Skipping copy.")
        return
    if not storage_path.is_dir():
        storage_path.mkdir(parents=True, exist_ok=True)
    source_file = Path(source_path, "subject_info.csv")
    target_file = Path(storage_path, "subject_info.csv")
    shutil.copy(str(source_file), str(target_file))


def get_samples_per_df(event_frames: Dict[str, pd.DataFrame], num_samples: int):
    """
    Get the number of samples per DataFrame based on the specified number of samples.

    Parameters
    ----------
    event_frames : Dict[str, pd.DataFrame]
        A dictionary where the keys are event types and the values are DataFrames containing event data.
    num_samples : int
        The total number of samples to distribute across the DataFrames.

    Returns
    -------
    sampled_dfs : Dict[str, pd.DataFrame]
        A dictionary with the same keys as `event_frames`, where each value is a sampled DataFrame.
    subject_events_per_df : Dict[str, int]
        A dictionary with the number of events per DataFrame.
    samples_per_df : Dict[str, int]
        A dictionary with the number of samples allocated to each DataFrame.
    """
    total_length = sum(len(df) for df in event_frames.values())
    samples_per_df = {
        event_types: int((len(df) / total_length) * num_samples)
        for event_types, df in event_frames.items()
    }

    # Adjust for rounding errors if necessary (simple method shown here)
    samples_adjusted = num_samples - sum(samples_per_df.values())
    for name in samples_per_df:
        if samples_adjusted <= 0:
            break
        samples_per_df[name] += 1
        samples_adjusted -= 1

    sampled_dfs = {
        event_types:
            event_frames[event_types][event_frames[event_types]["CHARTTIME"].isin(
                event_frames[event_types]["CHARTTIME"].unique()[:samples])]
            if len(event_frames[event_types]) >= samples else event_frames[event_types]
        for event_types, samples in samples_per_df.items()
    }

    subject_events_per_df = {
        event_types: len(samples) for event_types, samples in sampled_dfs.items()
    }

    if not sum([len(frames) for frames in sampled_dfs.values()]):
        raise RuntimeError(
            "Sample limit compliance subsampling produced empty dataframe. Source code is erroneous!"
        )

    return sampled_dfs, subject_events_per_df, samples_per_df


def convert_dtype_value(value, dtype: str):
    """
    Convert a value to the specified dtype.

    Parameters
    ----------
    value : any
        The value to be converted.
    dtype : str
        The target data type. Dtype can be one of "Int8", "Int16", "Int32", "Int64", "str", "float".

    Returns
    -------
    any
        The value converted to the specified data type.
    """
    dtype_mapping = {
        "Int8": np.int8,
        "Int16": np.int16,
        "Int32": np.int32,
        "Int64": np.int64,
        "str": str,
        "float": float,
        "float64": np.float64,
        "float32": np.float32,
        "object": lambda x: x
    }
    return dtype_mapping[dtype](value)


def convert_dtype_dict(dtypes: dict, add_lower=True) -> dict:
    """
    Convert a dictionary of column names to dtype strings to a dictionary of column names to dtype objects.

    Parameters
    ----------
    dtypes : dict
        Column name to dtype mapping. Dtype can be one of "Int8", "Int16", "Int32", "Int64", "str", "float".
    add_lower : bool, optional
        Whether to add lower case column names as well, by default True.

    Returns
    -------
    dict
        A dictionary with column name to dtype object mapping.
    """
    dtype_mapping = {
        "Int8": pd.Int8Dtype(),
        "Int16": pd.Int16Dtype(),
        "Int32": pd.Int32Dtype(),
        "Int64": pd.Int64Dtype(),
        "str": pd.StringDtype(),
        "float": float,
        "float64": pd.Float64Dtype(),
        "float32": pd.Float32Dtype(),
        "object": "object"
    }
    dtype_dict = {
        column: dtype_mapping[type_identifyer] for column, type_identifyer in dtypes.items()
    }
    if add_lower:
        dtype_dict.update({
            column.lower(): dtype_mapping[type_identifyer]
            for column, type_identifyer in dtypes.items()
        })
    return dtype_dict


def clean_chartevents_util(chartevents: pd.DataFrame):
    """
    Clean the chartevents DataFrame based on the timeseries column.

    Parameters
    ----------
    chartevents : pd.DataFrame
        DataFrame containing chart events.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    function_switch = DATASET_SETTINGS["CHARTEVENTS"]["clean"]
    for variable_name, function_identifier in function_switch.items():
        index = (chartevents.VARIABLE == variable_name)
        try:
            chartevents.loc[index, 'VALUE'] = globals()[function_identifier](chartevents.loc[index])
        except Exception as exp:
            print("Exception in clean_events function", function_identifier, ": ", exp)
            print("number of rows:", np.sum(index))
            print("values:", chartevents.loc[index])
            raise exp

    return chartevents.loc[chartevents.VALUE.notnull()]


def get_static_value(timeseries: pd.DataFrame, variable: str):
    """
    Get the first non-null value of a specified variable from the time series data.

    Parameters
    ----------
    timeseries : pd.DataFrame
        DataFrame containing the time series data.
    variable : str
        The variable to get the static value for.

    Returns
    -------
    any
        The first non-null value of the specified variable.
    """

    index = timeseries[variable].notnull()

    if index.any():
        loc = np.where(index)[0][0]
        return timeseries[variable].iloc[loc]

    return np.nan


def upper_case_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the column names to upper case for consistency.

    Parameters
    ----------
    frame : pd.DataFrame
        Target DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with upper-cased column names.
    """
    frame.columns = frame.columns.str.upper()

    return frame


def convert_to_numpy_types(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the dtypes to numpy types for consistency.

    Parameters
    ----------
    frame : pd.DataFrame
        Target DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with numpy dtypes.
    """
    for col in frame.columns:
        # Convert pandas "Int64" (or similar) types to "int64"
        if pd.api.types.is_integer_dtype(frame[col]):
            frame[col] = frame[col].astype('int64', errors='ignore')
        # Convert pandas "boolean" type to NumPy "bool"
        elif pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].astype('bool', errors='ignore')
        # Convert pandas "string" type to NumPy "object"
        elif pd.api.types.is_string_dtype(frame[col]):
            frame[col] = frame[col].astype('object', errors='ignore')
        # Convert pandas "Float64" (or similar) types to "float64"
        elif pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].astype('float64', errors='ignore')
    return frame


def _clean_height(df: pd.DataFrame) -> pd.Series:
    """
    Convert height from inches to centimeters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing height values.

    Returns
    -------
    pd.Series
        Series with converted height values.
    """
    value = df.VALUE.astype(float).copy()

    def get_measurment_type(string):
        """
        """
        return 'in' in string.lower()

    index = df.VALUEUOM.fillna('').apply(get_measurment_type) | df.MIMIC_LABEL.apply(
        get_measurment_type)
    value.loc[index] = np.round(value[index] * 2.54)
    return value


def _clean_systolic_bp(df: pd.DataFrame) -> pd.Series:
    """
    Filter out systolic blood pressure only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing blood pressure values.

    Returns
    -------
    pd.Series
        Series with systolic blood pressure values.
    """
    value = df.VALUE.astype(str).copy()
    index = value.apply(lambda string: '/' in string)
    value.loc[index] = value[index].apply(lambda string: re.match('^(\d+)/(\d+)$', string).group(1))
    return value.astype(float)


def _clean_diastolic_bp(df: pd.DataFrame) -> pd.Series:
    """
    Filter out diastolic blood pressure only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing blood pressure values.

    Returns
    -------
    pd.Series
        Series with diastolic blood pressure values.
    """
    value = df.VALUE.astype(str).copy()
    index = value.apply(lambda string: '/' in string)
    value.loc[index] = value[index].apply(lambda string: re.match('^(\d+)/(\d+)$', string).group(2))
    return value.astype(float)


def _clean_capilary_rr(df: pd.DataFrame) -> pd.Series:
    """
    Categorize capillary refill rate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing capillary refill rate values.

    Returns
    -------
    pd.Series
        Series with categorized capillary refill rate values.
    """
    df = df.copy()
    value = pd.Series(np.zeros(df.shape[0]), index=df.index).copy()
    value.loc[:] = np.nan

    df['VALUE'] = df.VALUE.astype(str)

    value.loc[(df.VALUE == 'Normal <3 secs') | (df.VALUE == 'Brisk')] = 0
    value.loc[(df.VALUE == 'Abnormal >3 secs') | (df.VALUE == 'Delayed')] = 1
    return value


def _clean_fraction_inspired_o2(df: pd.DataFrame) -> pd.Series:
    """
    Map fraction of inspired oxygen values to correct scale.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fraction of inspired oxygen values.

    Returns
    -------
    pd.Series
        Series with mapped fraction of inspired oxygen values.
    """
    value = df.VALUE.astype(float).copy()

    # Check wheather value is string
    is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=bool)

    def get_measurment_type(string):
        """
        torr is equal to mmHg
        """
        return 'torr' not in string.lower()

    index = df.VALUEUOM.fillna('').apply(get_measurment_type) & (is_str | (~is_str & (value > 1.0)))

    value.loc[index] = value[index] / 100.

    return value


def _clean_laboratory_values(df: pd.DataFrame) -> pd.Series:
    """
    Clean laboratory values by removing non-numeric entries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing laboratory values.

    Returns
    -------
    pd.Series
        Series with cleaned laboratory values.
    """
    value = df.VALUE.copy()
    index = value.apply(
        lambda string: type(string) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', string))
    value.loc[index] = np.nan
    return value.astype(float)


def _clean_o2sat(df: pd.DataFrame) -> pd.Series:
    """
    Scale oxygen saturation values to correct range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing oxygen saturation values.

    Returns
    -------
    pd.Series
        Series with scaled oxygen saturation values.
    """
    # change "ERROR" to NaN
    value = df.VALUE.copy()
    index = value.apply(
        lambda string: type(string) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', string))
    value.loc[index] = np.nan
    value = value.astype(float)

    # Scale values
    index = (value <= 1)
    value.loc[index] = value[index] * 100.
    return value


def _clean_temperature(df: pd.DataFrame) -> pd.Series:
    """
    Map Fahrenheit temperatures to Celsius.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing temperature values.

    Returns
    -------
    pd.Series
        Series with converted temperature values.
    """
    value = df.VALUE.astype(float).copy()

    def get_measurment_type(string):
        """
        """
        return 'F' in string

    index = df.VALUEUOM.fillna('').apply(get_measurment_type) | df.MIMIC_LABEL.apply(
        get_measurment_type) | (value >= 79)
    value.loc[index] = (value[index] - 32) * 5. / 9
    return value


def _clean_weight(df: pd.DataFrame) -> pd.Series:
    """
    Convert weight values to kilograms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing weight values.

    Returns
    -------
    pd.Series
        Series with converted weight values.
    """
    value = df.VALUE.astype(float).copy()

    def get_measurment_type(string):
        """
        """
        return 'oz' in string

    # ounces
    index = df.VALUEUOM.fillna('').apply(get_measurment_type) | df.MIMIC_LABEL.apply(
        get_measurment_type)
    value.loc[index] = value[index] / 16.

    def get_measurment_type(string):
        """
        """
        return 'lb' in string

    # pounds
    index = index | df.VALUEUOM.fillna('').apply(get_measurment_type) | df.MIMIC_LABEL.apply(
        get_measurment_type)
    value.loc[index] = value[index] * 0.453592
    return value


def _clean_respiratory_rate(df: pd.DataFrame) -> pd.Series:
    """
    Transform respiratory rate values from greater than 60 to 60.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing respiratory rate values.

    Returns
    -------
    pd.Series
        Series with cleaned respiratory rate values.
    """
    value = df.VALUE
    value = value.replace('>60/min retracts', 60)
    value = value.replace('>60/minute', 60)

    return value


def read_varmap_csv(resource_folder: Path):
    """
    Read the variable map CSV file from the specified resource folder.

    Parameters
    ----------
    resource_folder : Path
        The path to the resource folder containing the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the variable map.
    """
    csv_settings = DATASET_SETTINGS["varmap"]
    # Load the resource map
    varmap_df = pd.read_csv(Path(resource_folder, "itemid_to_variable_map.csv"),
                            index_col=None,
                            dtype=convert_dtype_dict(csv_settings["dtype"]))

    # Impute empty to string
    varmap_df = varmap_df.fillna('').astype(str)

    # Cast columns
    varmap_df['COUNT'] = varmap_df.COUNT.astype(int)
    varmap_df['ITEMID'] = varmap_df.ITEMID.astype(int)

    # Remove unlabeled and not occuring phenotypes and make sure only variables with ready status
    varmap_df = varmap_df.loc[(varmap_df['LEVEL2'] != '') & (varmap_df['COUNT'] > 0)]
    varmap_df = varmap_df.loc[(varmap_df.STATUS == 'ready')]

    # Get subdf
    varmap_df = varmap_df[['LEVEL2', 'ITEMID', 'MIMIC LABEL']].set_index('ITEMID')
    name_equivalences = {'LEVEL2': 'VARIABLE', 'MIMIC LABEL': 'MIMIC_LABEL'}
    varmap_df = varmap_df.rename(columns=name_equivalences)

    return varmap_df
