"""Dataset file

This file allows access to the dataset as specified.

Todo:


YerevaNN/mimic3-benchmarks
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
    Copy subject information from source path to storage path
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
    """_summary_

    Args:
        event_frames (Dict[str, pd.DataFrame]): _description_
        num_samples (int): _description_
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
        event_types: event_frames[event_types][event_frames[event_types]["CHARTTIME"].isin(
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
    """_summary_

    Args:
        dtypes (Dict[str, str]): column name to dtype maping. Dtype can be one of Int8, Int16, Int32, Int64, str, float" 

    Returns:
        dict: Returns dictionary with column name to dtype object mapping.
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


def make_writeboolean(storage_path: Path) -> bool:
    """_summary_

    Args:
        storage_path (Path): _description_

    Returns:
        bool: _description_
    """
    # TODO this is because i am unable to store and load a list within a dataframe
    # solve this
    if storage_path == None:
        return True

    if storage_path.name == "PHENO":
        return True

    if not storage_path.is_dir():
        os.mkdir(storage_path)
        return True
    subject_dirs = os.listdir(storage_path)
    if not subject_dirs:
        return True
    if not "episodic_data.csv" in os.listdir(Path(storage_path, subject_dirs.pop())):
        return True
    return False


def make_episodic_data(subject_icu_history: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        subject_icu_history (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Episodic data contains: Icustay, Age, Length of Stay, Mortality, Gender, Ethnicity, Height and Weight
    episodic_data = subject_icu_history[["ICUSTAY_ID", "AGE", "LOS",
                                         "MORTALITY"]].rename(columns={"ICUSTAY_ID": "Icustay"})

    def imputeby_map(string, map):
        """
        """
        if string in map:
            return map[string]
        return map['OTHER']

    # Impute gender
    episodic_data['GENDER'] = subject_icu_history.GENDER.fillna('').apply(
        imputeby_map, args=([DATASET_SETTINGS["gender_map"]]))

    # Impute ethnicity
    ethnicity_series = subject_icu_history.ETHNICITY.apply(
        lambda x: x.replace(' OR ', '/').split(' - ')[0].split('/')[0])
    episodic_data['ETHNICITY'] = ethnicity_series.fillna('').apply(
        imputeby_map, args=([DATASET_SETTINGS["ethnicity_map"]]))

    # Empty values
    episodic_data['Height'] = np.nan
    episodic_data['Weight'] = np.nan

    episodic_data = episodic_data.set_index('Icustay')

    return episodic_data


def make_diagnoses_util(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        diagnoses (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Diagnoese from each ICU stay with diagnose code in the column and stay ID as index
    diagnoses['VALUE'] = 1
    diagnoses = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates()
    labels = diagnoses.pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE')
    labels = labels.fillna(0).astype(int)
    labels = labels.reindex(columns=DATASET_SETTINGS["diagnosis_labels"])
    labels = labels.fillna(0).astype(int)

    return labels


def make_timeseries_util(chartevents: pd.DataFrame, variables) -> pd.DataFrame:
    """
    """
    # Lets create the time series
    metadata = chartevents[['CHARTTIME', 'ICUSTAY_ID']]
    metadata = metadata.sort_values(by=['CHARTTIME', 'ICUSTAY_ID'])
    metadata = metadata.drop_duplicates(keep='first').set_index('CHARTTIME')

    # Timeseries contains only the following. Subject_id and personal information in episodic data
    timeseries_df = chartevents[['CHARTTIME', 'VARIABLE', 'VALUE']]
    timeseries_df = timeseries_df.sort_values(by=['CHARTTIME', 'VARIABLE', 'VALUE'], axis=0)
    timeseries_df = timeseries_df.drop_duplicates(subset=['CHARTTIME', 'VARIABLE'], keep='last')
    timeseries_df = timeseries_df.pivot(index='CHARTTIME', columns='VARIABLE', values='VALUE')
    timeseries_df = timeseries_df.merge(metadata, left_index=True, right_index=True)
    timeseries_df = timeseries_df.sort_index(axis=0).reset_index()

    timeseries_df = timeseries_df.reindex(columns=np.append(variables, ['ICUSTAY_ID', 'CHARTTIME']))

    return timeseries_df


def make_episode(timeseries_df: pd.DataFrame,
                 stay_id: int,
                 intime=None,
                 outtime=None) -> pd.DataFrame:
    """
    """
    # All events with ID
    indices = (timeseries_df.ICUSTAY_ID == stay_id)

    # Plus all events int time frame
    if intime is not None and outtime is not None:
        indices = indices | ((timeseries_df.CHARTTIME >= intime) &
                             (timeseries_df.CHARTTIME <= outtime))

    # Filter out and remove ID (ID already in episodic data)
    timeseries_df = timeseries_df.loc[indices]
    del timeseries_df['ICUSTAY_ID']

    return timeseries_df


def make_hour_index(episode_df: pd.DataFrame,
                    intime,
                    remove_charttime: bool = True) -> pd.DataFrame:
    """
    """
    # Get difference and convert to hours
    episode_df = episode_df.copy()
    episode_df['hours'] = (episode_df.CHARTTIME -
                           intime).apply(lambda s: s / np.timedelta64(1, 's'))
    episode_df['hours'] = episode_df.hours / 60. / 60

    # Set index
    episode_df = episode_df.set_index('hours').sort_index(axis=0)

    if remove_charttime:
        del episode_df['CHARTTIME']

    return episode_df


def clean_chartevents_util(chartevents: pd.DataFrame):
    """
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
    """
    index = timeseries[variable].notnull()

    if index.any():
        loc = np.where(index)[0][0]
        return timeseries[variable].iloc[loc]

    return np.nan


def upper_case_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Converts the column names to upper case for consistency.

    Args:
        frame (pd.DataFrame): Target dataframe.

    Returns:
        pd.DataFrame: output dataframe.
    """
    frame.columns = frame.columns.str.upper()

    return frame


def convert_to_numpy_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Converts the dtypes to numpy types for consistency.

    Args:
        frame (pd.DataFrame): Target dataframe.

    Returns:
        pd.DataFrame: output dataframe.
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
    Convert inch to centimeter
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
    Filter out systolic blood preasure only. 
    """
    value = df.VALUE.astype(str).copy()
    index = value.apply(lambda string: '/' in string)
    value.loc[index] = value[index].apply(lambda string: re.match('^(\d+)/(\d+)$', string).group(1))
    return value.astype(float)


def _clean_diastolic_bp(df: pd.DataFrame) -> pd.Series:
    """
    Filter out diastolic blood preasure only. 
    """
    value = df.VALUE.astype(str).copy()
    index = value.apply(lambda string: '/' in string)
    value.loc[index] = value[index].apply(lambda string: re.match('^(\d+)/(\d+)$', string).group(2))
    return value.astype(float)


def _clean_capilary_rr(df: pd.DataFrame) -> pd.Series:
    """
    Categorize: Normal or Brisk: 0
                Abnormal or Delayed: 1
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
    many 0s, mapping 1<x<20 to 0<x<0.2 
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
    GLUCOSE, PH: sometimes have ERROR as value
    """
    value = df.VALUE.copy()
    index = value.apply(
        lambda string: type(string) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', string))
    value.loc[index] = np.nan
    return value.astype(float)


def _clean_o2sat(df: pd.DataFrame) -> pd.Series:
    """
    small number of 0<x<=1 that should be mapped to 0-100 scale
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
    map Farenheit to Celsius, some ambiguous 50<x<80
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
    Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
    Children are tough for height, weight
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
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    value = df.VALUE
    value = value.replace('>60/min retracts', 60)
    value = value.replace('>60/minute', 60)

    return value


def read_varmap_csv(resource_folder: Path):
    """
    Parameters:
        resource_folder:    If not default dataset path at data/mimic-iii-demo/resources/

    Returns:
        varmap_df:          Variable map containing relevant feature variables as provided by the benchmark code
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
