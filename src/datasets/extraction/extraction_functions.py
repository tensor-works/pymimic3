import pandas as pd
from utils.IO import *
from ..mimic_utils import *


def make_subject_events(chartevents_df: pd.DataFrame, icu_history_df: pd.DataFrame):
    """
    Parameters:
        chartevents_df:     Chartevent data from ICU bed
        icu_history_df:     ICU stay data

    Returns:
        subject_events:     Dictionary containing chartevents per subject ID
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


def make_timeseries(subject_events, subject_diagnoses, subject_icu_history, varmap_df):
    """
    Parameters:
        subject_events:         Chart and laboratory events per subject
        subject_diagnoses:      Diagnoses per ICU stay per patient 
        subject_icu_history:    ICU history per subject
        varmap_df:

    Returns:
        episodic_data:          Data describing each ICU stay 
        timeseries:             Timeseries of events over time for single ICU stay
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
        diagnosis_labels = make_diagnoses_util(current_diagnoses_df.reset_index(drop=True))
        episodic_data_df = make_episodic_data(current_icu_history_df)

        # Reset index before merge, so that we can keep it
        episodic_data_df = episodic_data_df.merge(diagnosis_labels,
                                                  left_index=True,
                                                  right_index=True)
        episodic_data_df.index.names = ["Icustay"]
        current_events_df = clean_chartevents_util(current_events_df)

        timeseries_df = make_timeseries_util(current_events_df, variables)

        timeseries[subject_id] = dict()

        for index in range(current_icu_history_df.shape[0]):
            stay_id = current_icu_history_df.ICUSTAY_ID.iloc[index]
            intime = current_icu_history_df.INTIME.iloc[index]
            outtime = current_icu_history_df.OUTTIME.iloc[index]
            icu_episode_df = make_episode(timeseries_df, stay_id, intime, outtime)

            if icu_episode_df.shape[0] == 0:
                continue

            icu_episode_df = make_hour_index(icu_episode_df, intime)

            episodic_data_df.loc[stay_id, 'Weight'] = get_static_value(icu_episode_df, 'Weight')
            episodic_data_df.loc[stay_id, 'Height'] = get_static_value(icu_episode_df, 'Height')

            timeseries[subject_id][stay_id] = icu_episode_df

        episodic_data[subject_id] = episodic_data_df

    return episodic_data, timeseries
