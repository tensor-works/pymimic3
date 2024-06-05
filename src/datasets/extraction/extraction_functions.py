"""
Extraction Functions Module
===========================

This module provides functions for extracting and processing ICU event data into a structured format.
The functions are used for both compact and iterative extraction processes. Compact extraction calls 
them from the __init__.py, while iterative calls them from the event_producer.py and timeseries_processor.py.

The primary output of these functions includes three main components:

1. **Subject Events**: A dictionary where each key is a subject ID, and the value is a DataFrame of 
   chart events (e.g., lab results, vital signs) associated with that subject.
        - **From**: CHARTEVENTS, LABEVENTS, OUTPUTEVENTS
        - **In**: evenet_consumer.py
        - **Cols**: 
            - SUBJECT_ID
            - HADM_ID
            - ICUSTAY_ID
            - CHARTTIME
            - ITEMID
            - VALUE
            - VALUEUOM

2. **Timeseries Data**: A dictionary where each key is a subject ID, and the value is another dictionary. 
   The inner dictionary maps each ICU stay ID to a DataFrame of time series data, containing recorded 
   events for specified variables (e.g., heart rate, blood pressure) indexed by time.
    - **From**: icu history, subject events, varmap
    - **In**: timeseries_processor.py
    - **Cols**: 
            - Capillary refill rate
            - Diastolic blood pressure
            - Fraction inspired oxygen
            - Glascow coma scale eye opening
            - Glascow coma scale motor response
            - Glascow coma scale total
            - Glascow coma scale verbal response
            - Glucose
            - Heart Rate
            - Height
            - Mean blood pressure
            - Oxygen saturation
            - pH
            - Respiratory rate
            - Systolic blood pressure
            - Temperature
            - Weight

3. **Episodic Data**: A dictionary where each key is a subject ID, and the value is a DataFrame of 
   episodic data, summarizing each ICU stay with patient-specific and stay-specific information 
   (e.g., age, length of stay, mortality, gender, ethnicity, height, weight, and diagnoses).
        - **From**: icu history, subject events, diagnoses
        - **In**: timeseries_processor.py
        - **Cols**:
            - ICU Stay ID
            - Age
            - Length of Stay (LOS)
            - Mortality
            - Gender
            - Ethnicity
            - Height
            - Weight
            - Diagnoses binaries as columns
"""

import pandas as pd
from utils.IO import *
from ..mimic_utils import *

__all__ = ["extract_subject_events", "extract_timeseries", "extract_episodic_data"]


def extract_subject_events(chartevents_df: pd.DataFrame, icu_history_df: pd.DataFrame):
    """
    Extracts and processes chartevent data into a dictionary of events by subject.

    This method processes the input DataFrame containing chartevent from the OUTPUTEVENTS, LABEVENTS or INPUTEVENTS CSV's 
    and merges it with the ICU stay history data to create a structured dictionary of events per subject and stay ID. 
    It filters and aligns the chartevents with the corresponding ICU stays, ensuring that each event is correctly associated with 
    the appropriate ICU stay and subject, as some events might not include these IDs. If not ICUSTAY_ID could be found,
    the data is dropped.

    Parameters
    ----------
    chartevents_df : pd.DataFrame
        DataFrame containing chartevent data from ICU.
    icu_history_df : pd.DataFrame
        DataFrame containing ICU stay data.

    Returns
    -------
    dict
        Dictionary containing chartevents per subject ID.
    """
    chartevents_df = chartevents_df.dropna(subset=['HADM_ID'])
    recovered_df = chartevents_df.merge(icu_history_df,
                                        left_on=['HADM_ID'],
                                        right_on=['HADM_ID'],
                                        how='left',
                                        suffixes=['', '_r'],
                                        indicator=True)
    recovered_df = recovered_df[recovered_df['_merge'] == 'both']
    recovered_df['ICUSTAY_ID'] = recovered_df['ICUSTAY_ID'].fillna(recovered_df['ICUSTAY_ID_r'])
    recovered_df = recovered_df.dropna(subset=['ICUSTAY_ID'])
    recovered_df = recovered_df[(recovered_df['ICUSTAY_ID'] == recovered_df['ICUSTAY_ID_r'])]
    recovered_df = recovered_df[[
        'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'
    ]]

    return {id: x for id, x in recovered_df.groupby('SUBJECT_ID') if not x.empty}


def extract_timeseries(subject_events, subject_diagnoses, subject_icu_history, varmap_df):
    """
    Processes subject events, diagnoses, and ICU history into time series and episodic data.

    This method retrieves the ICU history, diagnoses, and events data for each subject. It merges the events data 
    with variable mappings, sorts the ICU stays by admission and discharge times, and extracts diagnosis labels and 
    general patient data for each ICU stay. These are then stored in the episodic_data_df. The method then cleans and processes the events data
     (e.g. convert to same metric units, remove null values), 
    extracts time series data for each variable of interest, per episode. It extracts hourly indexed time 
    series data and retrieves static values such as weight and height for each ICU stay. Finally, the episodic and timeseries
    data are stored in the subject directory. 

    **Episodic data** includes the following information for each ICU stay:
        - ICU Stay ID
        - Age
        - Length of Stay (LOS)
        - Mortality
        - Gender
        - Ethnicity
        - Height
        - Weight
        - Diagnoses binaries as columns
    
    **Time series** variables of interest are: 
        - Capillary refill rate
        - Diastolic blood pressure
        - Fraction inspired oxygen
        - Glascow coma scale eye opening
        - Glascow coma scale motor response
        - Glascow coma scale total
        - Glascow coma scale verbal response
        - Glucose
        - Heart Rate
        - Height
        - Mean blood pressure
        - Oxygen saturation
        - pH
        - Respiratory rate
        - Systolic blood pressure
        - Temperature
        - Weight

    Parameters
    ----------
    subject_events : dict
        Dictionary containing chart and laboratory events per subject.
    subject_diagnoses : dict
        Dictionary containing diagnoses per ICU stay per patient.
    subject_icu_history : dict
        Dictionary containing ICU history per subject.
    varmap_df : pd.DataFrame
        DataFrame containing variable mappings.

    Returns
    -------
    tuple
        - episodic_data : dict
            Dictionary describing each ICU stay.
        - timeseries : dict
            Dictionary containing time series of events over time for each ICU stay.
    """
    variables = varmap_df.VARIABLE.unique()
    timeseries = dict()
    episodic_data = dict()

    for subject_id in subject_icu_history.keys():
        try:
            current_icu_history_df: pd.DataFrame = subject_icu_history[subject_id]
            current_diagnoses_df: pd.DataFrame = subject_diagnoses[subject_id]
            current_events_df: pd.DataFrame = subject_events[subject_id]
        except:
            continue

        # Adjust current events
        current_events_df = current_events_df.merge(varmap_df, left_on='ITEMID', right_index=True)
        current_events_df = current_events_df.loc[current_events_df.VALUE.notnull()]
        current_events_df.loc[:, 'VALUEUOM'] = current_events_df['VALUEUOM'].fillna('').astype(str)

        # Sort stays
        current_icu_history_df = current_icu_history_df.sort_values(by=['INTIME', 'OUTTIME'])

        # General patient data belonging to each ICU stay
        diagnosis_labels = extract_diagnoses_util(current_diagnoses_df.reset_index(drop=True))
        episodic_data_df = extract_episodic_data(current_icu_history_df)

        # Reset index before merge, so that we can keep it
        episodic_data_df = episodic_data_df.merge(diagnosis_labels,
                                                  left_index=True,
                                                  right_index=True)
        episodic_data_df.index.names = ["Icustay"]
        current_events_df = clean_chartevents_util(current_events_df)

        timeseries_df = extract_timeseries_util(current_events_df, variables)

        timeseries[subject_id] = dict()

        for index in range(current_icu_history_df.shape[0]):
            stay_id = current_icu_history_df.ICUSTAY_ID.iloc[index]
            intime = current_icu_history_df.INTIME.iloc[index]
            outtime = current_icu_history_df.OUTTIME.iloc[index]
            icu_episode_df = extract_episode(timeseries_df, stay_id, intime, outtime)

            if icu_episode_df.shape[0] == 0:
                continue

            icu_episode_df = extract_hour_index(icu_episode_df, intime)

            episodic_data_df.loc[stay_id, 'Weight'] = get_static_value(icu_episode_df, 'Weight')
            episodic_data_df.loc[stay_id, 'Height'] = get_static_value(icu_episode_df, 'Height')

            timeseries[subject_id][stay_id] = icu_episode_df

        episodic_data[subject_id] = episodic_data_df

    return episodic_data, timeseries


def extract_episodic_data(subject_icu_history: pd.DataFrame) -> pd.DataFrame:
    """
    Create episodic data from subject ICU history.

    This method processes the ICU history of subjects to create a DataFrame containing episodic data for each ICU stay.
    The episodic data includes per ICU stay:
    - ID
    - age
    - length of stay
    - mortality
    - gender
    - ethnicity
    - height
    - weight

    Gender and ethnicity are imputed using predefined mappings if they are missing.

    Parameters
    ----------
    subject_icu_history : pd.DataFrame
        DataFrame containing the subject ICU history.

    Returns
    -------
    pd.DataFrame
        DataFrame containing episodic data.
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


def extract_diagnoses_util(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame of diagnoses from each ICU stay with diagnose code in the column and stay ID as index.

    This method processes the diagnoses DataFrame to create a binary matrix indicating the presence of each diagnosis 
    present in the diagnoses.csv, as code over ICU stay. The resulting DataFrame has ICU stay IDs as rows and diagnosis codes as columns, with 
    binary values indicating whether each diagnosis code is present.

    Parameters
    ----------
    diagnoses : pd.DataFrame
        DataFrame containing diagnoses.

    Returns
    -------
    pd.DataFrame
        DataFrame with diagnoses as columns and stay IDs as rows.
    """
    # Diagnoese from each ICU stay with diagnose code in the column and stay ID as index
    diagnoses['VALUE'] = 1
    diagnoses = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates()
    labels = diagnoses.pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE')
    labels = labels.fillna(0).astype(int)
    labels = labels.reindex(columns=DATASET_SETTINGS["diagnosis_labels"])
    labels = labels.fillna(0).astype(int)

    return labels


def extract_timeseries_util(subject_events: pd.DataFrame, variables) -> pd.DataFrame:
    """
    This method processes the input DataFrame containing chart events to generate a time series DataFrame for the 
    specified variables. The data is pivoted against the VARIABLE (17 main features), which are now the columns containing 
    the entry from the previous VALUE row, so that each row represents a specific chart time. The resulting DataFrame 
    contains the time series data of each variable indexed by chart time and includes the ICU stay ID.

    Parameters
    ----------
    chartevents : pd.DataFrame
        DataFrame containing chart events.
    variables : list
        List of variables to include in the time series.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the time series data.
    """
    # Lets create the time series
    metadata = subject_events[['CHARTTIME', 'ICUSTAY_ID']]
    metadata = metadata.sort_values(by=['CHARTTIME', 'ICUSTAY_ID'])
    metadata = metadata.drop_duplicates(keep='first').set_index('CHARTTIME')

    # Timeseries contains only the following. Subject_id and personal information in episodic data
    timeseries_df = subject_events[['CHARTTIME', 'VARIABLE', 'VALUE']]
    timeseries_df = timeseries_df.sort_values(by=['CHARTTIME', 'VARIABLE', 'VALUE'], axis=0)
    timeseries_df = timeseries_df.drop_duplicates(subset=['CHARTTIME', 'VARIABLE'], keep='last')
    timeseries_df = timeseries_df.pivot(index='CHARTTIME', columns='VARIABLE', values='VALUE')
    timeseries_df = timeseries_df.merge(metadata, left_index=True, right_index=True)
    timeseries_df = timeseries_df.sort_index(axis=0).reset_index()

    timeseries_df = timeseries_df.reindex(columns=np.append(variables, ['ICUSTAY_ID', 'CHARTTIME']))

    return timeseries_df


def extract_episode(timeseries_df: pd.DataFrame,
                    stay_id: int,
                    intime=None,
                    outtime=None) -> pd.DataFrame:
    """
    Create an episode DataFrame from the time series data.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        DataFrame containing the time series data.
    stay_id : int
        The stay ID to filter the time series data.
    intime : datetime, optional
        The admission time to filter the time series data.
    outtime : datetime, optional
        The discharge time to filter the time series data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the episode data.
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


def extract_hour_index(episode_df: pd.DataFrame,
                       intime,
                       remove_charttime: bool = True) -> pd.DataFrame:
    """
    Create an hour-based index for the episode DataFrame.

    Parameters
    ----------
    episode_df : pd.DataFrame
        DataFrame containing the episode data.
    intime : datetime
        The admission time to calculate the hours.
    remove_charttime : bool, optional
        Whether to remove the CHARTTIME column, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with an hour-based index.
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
