import pandas as pd
import yaml
import warnings
import random

warnings.simplefilter(action='ignore', category=FutureWarning)

from copy import deepcopy
from pathlib import Path
from typing import List
from utils.IO import *
from settings import *
from .extraction_functions import make_subject_events, make_timeseries
from .event_producer import EventProducer
from .timeseries_processor import TimeseriesProcessor
from ..trackers import ExtractionTracker
from ..mimic_utils import *
from ..readers import ExtractedSetReader, EventReader
from ..writers import DataSetWriter

__all__ = ["compact_extraction", "iterative_extraction"]


def compact_extraction(storage_path: Path,
                       source_path: Path,
                       num_subjects: int = None,
                       num_samples: int = None,
                       subject_ids: list = None,
                       task: str = None):
    """_summary_

    Args:
        source_path (_type_, optional): _description_. Defaults to None.
        ehr (_type_, optional): _description_. Defaults to None.
        from_storage (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    original_subject_ids = deepcopy(subject_ids)
    resource_folder = Path(source_path, "resources")
    tracker = ExtractionTracker(storage_path=Path(storage_path, "progress"),
                                num_subjects=num_subjects,
                                num_samples=num_samples,
                                subject_ids=subject_ids)

    dataset_writer = DataSetWriter(storage_path)
    dataset_reader = ExtractedSetReader(storage_path)

    if tracker.is_finished:  # and not compute_data:
        info_io(f"Compact data extraction already finalized in directory:\n{str(storage_path)}")
        if task is not None:
            # Make sure we don't pick empty subjects for the subsequent processing
            icu_history_df = dataset_reader.read_csv(Path(storage_path, "icu_history.csv"),
                                                     dtypes=convert_dtype_dict(
                                                         DATASET_SETTINGS["icu_history"]["dtype"]))

            subject_ids = get_processable_subjects(task, icu_history_df)
            subject_ids = list(set(subject_ids) & set(tracker.subject_ids))
        else:
            subject_ids = tracker.subject_ids

        if num_subjects is not None and num_subjects < len(subject_ids):
            subject_ids = random.sample(subject_ids, k=num_subjects)
        elif original_subject_ids is not None:
            subject_ids = set(original_subject_ids) & set(subject_ids)
        # If we know some processing is comming after we return all possible subjects for that task
        return dataset_reader.read_subjects(read_ids=True, subject_ids=subject_ids)

    info_io("Compact Dataset Extraction: ALL", level=0)
    info_io(f"Starting compact data extraction.")
    info_io(f"Extracting data from source:\n{str(source_path)}")
    info_io(f"Saving data at location:\n{str(storage_path)}")

    # Read Dataframes for ICU history
    if tracker.has_icu_history:
        info_io("ICU history data already extracted")
        icu_history_df = dataset_reader.read_csv(Path(storage_path, "icu_history.csv"),
                                                 dtypes=convert_dtype_dict(
                                                     DATASET_SETTINGS["icu_history"]["dtype"]))
        subject_info_df = dataset_reader.read_csv(Path(storage_path, "subject_info.csv"),
                                                  dtypes=convert_dtype_dict(
                                                      DATASET_SETTINGS["subject_info"]["dtype"]))
    else:
        info_io("Extracting ICU history data")
        patients_df = read_patients_csv(source_path)
        admissions_df, admissions_info_df = read_admission_csv(source_path)
        icustays_df = read_icustays_csv(source_path)

        # Make ICU history dataframe
        subject_info_df = make_subject_infos(patients_df, admissions_info_df, icustays_df)
        icu_history_df = make_icu_history(patients_df, admissions_df, icustays_df)
        tracker.has_icu_history = True
        icu_history_df.to_csv(Path(storage_path, "icu_history.csv"), index=False)
        subject_info_df.to_csv(Path(storage_path, "subject_info.csv"), index=False)

    if tracker.has_diagnoses:
        info_io("Patient diagnosis data already extracted")
        diagnoses_df = dataset_reader.read_csv(Path(storage_path, "diagnoses.csv"),
                                               dtypes=convert_dtype_dict(
                                                   DATASET_SETTINGS["diagnosis"]["dtype"]))
    else:
        info_io("Extracting patient diagnosis data")
        # Read Dataframes for diagnoses
        icd9codes_df = read_icd9codes_csv(source_path)

        #
        diagnoses_df, definition_map = make_diagnoses(source_path, icd9codes_df, icu_history_df)
        diagnoses_df.to_csv(Path(storage_path, "diagnoses.csv"), index=False)

        tracker.has_diagnoses = True

    subject_ids, icu_history_df = get_subject_ids(task=task,
                                                  num_subjects=num_subjects,
                                                  subject_ids=subject_ids,
                                                  existing_subjects=tracker.subject_ids,
                                                  icu_history_df=icu_history_df)

    subject_info_df = reduce_by_subjects(subject_info_df, subject_ids)
    diagnoses_df = reduce_by_subjects(diagnoses_df, subject_ids)

    info_io("Extracting subject ICU history")
    subject_icu_history = get_by_subject(icu_history_df,
                                         DATASET_SETTINGS["ICUHISTORY"]["sort_value"])

    info_io("Extracting subject diagnoses")
    subject_diagnoses = get_by_subject(diagnoses_df[DATASET_SETTINGS["DIAGNOSES"]["columns"]],
                                       DATASET_SETTINGS["DIAGNOSES"]["sort_value"])
    if tracker.has_bysubject_info:
        info_io("Subject diagnoses and subject ICU history already stored")
    else:
        dataset_writer.write_bysubject(
            {
                "subject_icu_history": subject_icu_history,
                "subject_diagnoses": subject_diagnoses
            },
            index=False)

    if tracker.has_subject_events:
        info_io("Subject events already extracted")
        subject_events = dataset_reader.read_events(read_ids=True)
    else:
        info_io("Extracting subject events")
        # Read Dataframes for event table
        event_reader = EventReader(source_path)
        chartevents_df = event_reader.get_all()

        # Make subject event table
        subject_events = make_subject_events(chartevents_df, icu_history_df)
        dataset_writer.write_bysubject({
            "subject_events": subject_events,
        }, index=False)
        tracker.has_subject_events = True

    if not tracker.has_timeseries or not tracker.has_episodic_data:
        info_io("Extracting subject timeseries and episodic data")
        # Read Dataframes for time series
        varmap_df = read_varmap_csv(resource_folder)
        episodic_data, timeseries = make_timeseries(subject_events, subject_diagnoses,
                                                    subject_icu_history, varmap_df)
        name_data_pair = {"episodic_data": episodic_data, "timeseries": timeseries}
        dataset_writer.write_bysubject(name_data_pair, exists_ok=True)

        tracker.subject_ids.extend(list(timeseries.keys()))
        tracker.has_episodic_data = True
        tracker.has_timeseries = True
    else:
        info_io("Subject timeseries and episodic data already extracted")

    tracker.is_finished = True
    info_io(f"Finalized data extraction in directory:\n{str(storage_path)}")
    if original_subject_ids is not None:
        original_subject_ids = list(set(original_subject_ids) & set(tracker.subject_ids))
    return dataset_reader.read_subjects(read_ids=True, subject_ids=original_subject_ids)


def iterative_extraction(source_path: Path,
                         storage_path: Path = None,
                         chunksize: int = None,
                         num_subjects: int = None,
                         num_samples: int = None,
                         subject_ids: list = None,
                         task: str = None):
    """_summary_

    Args:
        source_path (_type_): _description_
        source_path (_type_): _description_
        ehr (_type_): _description_
        from_storage (_type_): _description_
        chunksize (_type_): _description_
        num_subjects (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Not sure
    original_subject_ids = deepcopy(subject_ids)
    resource_folder = Path(source_path, "resources")

    tracker = ExtractionTracker(storage_path=Path(storage_path, "progress"),
                                num_samples=num_samples,
                                num_subjects=num_subjects,
                                subject_ids=subject_ids)

    dataset_writer = DataSetWriter(storage_path)
    dataset_reader = ExtractedSetReader(source_path)

    if tracker.is_finished:
        info_io(f"Iterative data extraction already finalized in directory:\n{storage_path}.")
        if task is not None:
            # Make sure we don't pick empty subjects for the subsequent processing
            icu_history_df = dataset_reader.read_csv(Path(storage_path, "icu_history.csv"),
                                                     dtypes=convert_dtype_dict(
                                                         DATASET_SETTINGS["icu_history"]["dtype"]))

            subject_ids = get_processable_subjects(task, icu_history_df)
            subject_ids = list(set(subject_ids) & set(tracker.subject_ids))
        else:
            subject_ids = tracker.subject_ids

        if num_subjects is not None and num_subjects < len(subject_ids):
            subject_ids = random.sample(subject_ids, k=num_subjects)
        elif original_subject_ids is not None:
            subject_ids = set(original_subject_ids) & set(subject_ids)
        # If we know some processing is comming after we return all possible subjects for that task
        return ExtractedSetReader(storage_path, subject_ids=subject_ids)

    info_io("Iterative Dataset Extraction: ALL", level=0)
    info_io(f"Starting iterative data extraction.")
    info_io(f"Extracting data from source:\n{str(source_path)}")
    info_io(f"Saving data at location:\n{str(storage_path)}")

    # Make ICU history dataframe
    if tracker.has_icu_history:
        info_io("ICU history data already extracted")
        icu_history_df = dataset_reader.read_csv(Path(storage_path, "icu_history.csv"),
                                                 dtypes=convert_dtype_dict(
                                                     DATASET_SETTINGS["icu_history"]["dtype"]))
        subject_info_df = dataset_reader.read_csv(Path(storage_path, "subject_info.csv"),
                                                  dtypes=convert_dtype_dict(
                                                      DATASET_SETTINGS["subject_info"]["dtype"]))
    else:
        # Read Dataframes for ICU history
        info_io("Extracting ICU history data")
        patients_df = read_patients_csv(source_path)

        admissions_df, admission_info_df = read_admission_csv(source_path)
        icustays_df = read_icustays_csv(source_path)
        # Generate history
        subject_info_df = make_subject_infos(patients_df, admission_info_df, icustays_df)
        icu_history_df = make_icu_history(patients_df, admissions_df, icustays_df)

        # TODO! encapsualte in function
        icu_history_df.to_csv(Path(storage_path, "icu_history.csv"), index=False)
        subject_info_df.to_csv(Path(storage_path, "subject_info.csv"), index=False)
        tracker.has_icu_history = True

    # Read Dataframes for diagnoses

    if tracker.has_diagnoses:
        info_io("Patient diagnosis data already extracted")
        diagnoses_df = dataset_reader.read_csv(Path(storage_path, "diagnoses.csv"),
                                               dtypes=convert_dtype_dict(
                                                   DATASET_SETTINGS["diagnosis"]["dtype"]))
    else:
        info_io("Extracting Patient diagnosis data")

        icd9codes_df = read_icd9codes_csv(source_path)
        diagnoses_df, definition_map = make_diagnoses(source_path, icd9codes_df, icu_history_df)
        # TODO! not working
        # make_phenotypes(diagnoses_df,
        #                definition_map).to_csv(Path(storage_path, "phenotype_matrix.csv"))
        diagnoses_df.to_csv(Path(storage_path, "diagnoses.csv"), index=False)
        tracker.has_diagnoses = True

    subject_ids, icu_history_df = get_subject_ids(task=task,
                                                  num_subjects=num_subjects,
                                                  subject_ids=subject_ids,
                                                  existing_subjects=tracker.subject_ids,
                                                  icu_history_df=icu_history_df)

    diagnoses_df = reduce_by_subjects(diagnoses_df, subject_ids)
    subject_diagnoses = get_by_subject(diagnoses_df[DATASET_SETTINGS["DIAGNOSES"]["columns"]],
                                       DATASET_SETTINGS["DIAGNOSES"]["sort_value"])
    subject_icu_history = get_by_subject(icu_history_df,
                                         DATASET_SETTINGS["ICUHISTORY"]["sort_value"])
    if not tracker.has_subject_events:
        name_data_pairs = {
            "subject_diagnoses": {
                subject_id: frame_df for subject_id, frame_df in subject_diagnoses.items()
            },
            "subject_icu_history": {
                subject_id: frame_df for subject_id, frame_df in subject_icu_history.items()
            }
        }
        dataset_writer.write_bysubject(name_data_pairs, index=False)
    else:
        info_io("Subject diagnoses and subject ICU history already stored")

    if not tracker.has_subject_events:
        info_io("Extracting subject events")

        EventProducer(source_path=source_path,
                      storage_path=storage_path,
                      num_samples=num_samples,
                      chunksize=chunksize,
                      tracker=tracker,
                      icu_history_df=icu_history_df,
                      subject_ids=subject_ids).run()
    else:
        info_io("Subject events already extracted")

    varmap_df = read_varmap_csv(resource_folder)

    if not tracker.has_episodic_data or not tracker.has_timeseries:
        info_io("Extraction timeseries data from subject events")

        # Starting the processor pool
        pool_processor = TimeseriesProcessor(storage_path=storage_path,
                                             source_path=source_path,
                                             tracker=tracker,
                                             subject_ids=subject_ids,
                                             diagnoses_df=subject_diagnoses,
                                             icu_history_df=subject_icu_history,
                                             varmap_df=varmap_df,
                                             num_samples=num_samples)

        pool_processor.run()
        info_io(f"Subject directories extracted: {len(tracker.subject_ids)}")

        tracker.has_episodic_data = True
        tracker.has_timeseries = True
    else:
        info_io(f"Timeseries data already created")

    tracker.is_finished = True
    if original_subject_ids is not None:
        original_subject_ids = list(set(original_subject_ids) & set(tracker.subject_ids))
    return ExtractedSetReader(storage_path, subject_ids=original_subject_ids)


def reduce_by_subjects(dataframe: pd.DataFrame, subject_ids: list):
    if subject_ids is not None:
        return dataframe[dataframe["SUBJECT_ID"].isin(subject_ids)]
    return dataframe


def get_subject_ids(task: str,
                    icu_history_df: pd.DataFrame,
                    subject_ids: list = None,
                    num_subjects: int = None,
                    existing_subjects: list = None):
    """Get the subject IDs to be processed or returned.

    Args:
        task (str): _description_
        icu_history_df (pd.DataFrame): _description_
        subject_ids (list, optional): _description_. Defaults to None.
        num_subjects (int, optional): _description_. Defaults to None.
        existing_subjects (list, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if existing_subjects is None:
        existing_subjects = []

    if subject_ids is not None:
        all_subjects = get_processable_subjects(task, icu_history_df)
        # Notify unknowns
        unknown_subjects = set(subject_ids) - set(icu_history_df["SUBJECT_ID"].unique())
        if unknown_subjects:
            warn_io(f"Unknown subjects passed as parameter: {*unknown_subjects,}")
        # Notify unprocessable
        unprocessable_subjects = set(subject_ids) - set(all_subjects)
        if unprocessable_subjects:
            warn_io(f"Unprocessable subjects passed as parameter: {*unprocessable_subjects,}")
        # Remove already processed
        subject_ids = list((set(subject_ids) - set(existing_subjects)) & set(all_subjects))
        icu_history_df = reduce_by_subjects(icu_history_df, subject_ids)
    elif num_subjects is not None:
        all_subjects = get_processable_subjects(task, icu_history_df)
        subject_ids = get_subjects_by_number(task, num_subjects, existing_subjects, all_subjects)
        icu_history_df = reduce_by_subjects(icu_history_df, subject_ids)
    else:
        subject_ids = None
    return subject_ids, icu_history_df


def get_subjects_by_number(task: str, num_subjects: int, existing_subjects: List[int],
                           all_subjects: List[int]):
    # Determined how many are missing
    num_subjects = max(0, num_subjects - len(existing_subjects))
    # Chose from uprocessed subjects
    if num_subjects > len(all_subjects):
        raise warn_io(
            f"Number of subjects requested exceeds available subjects: {len(all_subjects)}")
    remaining_subjects = list(set(all_subjects) - set(existing_subjects))
    # if tracker is None we grab all possible subjects for return to the next processing step
    subject_ids = random.sample(remaining_subjects, k=num_subjects)
    assert len(subject_ids) == num_subjects
    return subject_ids


def get_processable_subjects(task: str, icu_history_df: pd.DataFrame):
    if task is not None and "label_start_time" in DATASET_SETTINGS[task]:
        # Some ids will be removed during the preprocessing step
        # We remove them here to avoid errors
        min_los = DATASET_SETTINGS[task]["label_start_time"] + \
            DATASET_SETTINGS[task]["sample_precision"]
        min_los /= 24
        icu_history_df = icu_history_df[icu_history_df["LOS"] >= min_los]
        return icu_history_df[((icu_history_df["DISCHTIME"] - icu_history_df["ADMITTIME"])
                               >= pd.Timedelta(days=min_los))]["SUBJECT_ID"].unique()
    else:
        return icu_history_df["SUBJECT_ID"].unique()


def create_split_info_csv(episodic_info_df: pd.DataFrame, subject_info_df: pd.DataFrame):
    """_summary_

    Args:
        episodic_data_df (pd.DataFrame): _description_
        icu_history_df (pd.DataFrame): _description_
        events_info_df (pd.DataFrame): _description_
    """
    episodic_info_df["SUBJECT_ID"] = episodic_info_df["SUBJECT_ID"].astype(int)
    episodic_info_df = episodic_info_df.merge(subject_info_df,
                                              how='inner',
                                              left_on=['SUBJECT_ID', 'ICUSTAY_ID'],
                                              right_on=['SUBJECT_ID', 'ICUSTAY_ID'])
    episodic_info_df.to_csv("split_info.csv")

    return


def read_patients_csv(dataset_folder: Path):
    """
    Parameters:
        dataset_folder:     If not default dataset path at data/mimic-iii-demo/

    Returns:
        patients_df:        Patients data (birth, death, gender, ethnicity etc.)
    """
    csv_settings = DATASET_SETTINGS["PATIENTS"]
    patients_df = pd.read_csv(Path(dataset_folder, "PATIENTS.csv"),
                              dtype=convert_dtype_dict(csv_settings["dtype"]))

    patients_df = upper_case_column_names(patients_df)
    patients_df = patients_df[csv_settings["columns"]].copy()

    for column in csv_settings["convert_datetime"]:
        patients_df[column] = pd.to_datetime(patients_df[column])

    return patients_df


def read_admission_csv(dataset_folder: Path):
    """
    Parameters:
        dataset_folder:     If not default dataset path at data/mimic-iii-demo/

    Returns:
        admissions_df:      Hospital admissions data (admission, discharge, type, location, etc.)
    """
    csv_settings = DATASET_SETTINGS["ADMISSIONS"]

    admissions_df = pd.read_csv(Path(dataset_folder, "ADMISSIONS.csv"),
                                dtype=convert_dtype_dict(csv_settings["dtype"]))
    admissions_df = upper_case_column_names(admissions_df)
    for column in csv_settings["convert_datetime"]:
        admissions_df[column] = pd.to_datetime(admissions_df[column])

    admissions_info_df = admissions_df[csv_settings["info_columns"]]
    admissions_df = admissions_df[csv_settings["columns"]].copy()

    return admissions_df, admissions_info_df


def read_icustays_csv(dataset_folder: Path):
    """
    Parameters:
        dataset_folder:     If not default dataset path at data/mimic-iii-demo/

    Returns:
        icustays_df:        ICU admission data (firt & last care unit, ward id, etc.)
    """
    csv_settings = DATASET_SETTINGS["ICUSTAYS"]

    icustays_df = pd.read_csv(Path(dataset_folder, "ICUSTAYS.csv"),
                              dtype=convert_dtype_dict(csv_settings["dtype"]))
    icustays_df = upper_case_column_names(icustays_df)
    for column in csv_settings["convert_datetime"]:
        icustays_df[column] = pd.to_datetime(icustays_df[column])

    return icustays_df


def read_icd9codes_csv(dataset_folder: Path):
    """
    Parameters:
        dataset_folder:     If not default dataset path at data/mimic-iii-demo/

    Returns:
        icd9codes_df:       Dictionaries describing the ICD9 codes
    """
    csv_settings = DATASET_SETTINGS["ICD9CODES"]

    icd9codes_df = pd.read_csv(Path(dataset_folder, 'D_ICD_DIAGNOSES.csv'),
                               dtype=convert_dtype_dict(csv_settings["dtype"]))
    icd9codes_df = upper_case_column_names(icd9codes_df)
    icd9codes_df = icd9codes_df[csv_settings["columns"]]

    return icd9codes_df


def read_events_dictionary(dataset_folder: Path):
    """_summary_

    Args:
        dataset_folder (_type_): _description_
    """
    csv_settings = DATASET_SETTINGS["D_ITEMS"]
    dictionary_df = pd.read_csv(Path(dataset_folder, "D_ITEMS.csv"),
                                dtype=convert_dtype_dict(csv_settings["dtype"]))
    dictionary_df = upper_case_column_names(dictionary_df)
    dictionary_df = dictionary_df[["ITEMID", "DBSOURCE"]]

    return dictionary_df


def merge_patient_history(patients_df, admissions_df, icustays_df, min_nb_stays,
                          max_nb_stays) -> pd.DataFrame:
    """_summary_

    Args:
        patients_df (_type_): _description_
        admissions_df (_type_): _description_
        icustays_df (_type_): _description_
        min_nb_stays (_type_): _description_
        max_nb_stays (_type_): _description_
    """
    icustays_df = icustays_df[icustays_df["FIRST_CAREUNIT"] == icustays_df["LAST_CAREUNIT"]]
    icustays_df = icustays_df[icustays_df["FIRST_WARDID"] == \
                                    icustays_df["LAST_WARDID"]]

    patient_history_df = icustays_df.merge(admissions_df,
                                           how='inner',
                                           left_on=['SUBJECT_ID', 'HADM_ID'],
                                           right_on=['SUBJECT_ID', 'HADM_ID'])
    patient_history_df = patient_history_df.merge(patients_df,
                                                  how='inner',
                                                  left_on=['SUBJECT_ID'],
                                                  right_on=['SUBJECT_ID'])

    filter = patient_history_df.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    filter = filter.loc[(filter.ICUSTAY_ID >= min_nb_stays)
                        & (filter.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    patient_history_df = patient_history_df.merge(filter,
                                                  how='inner',
                                                  left_on='HADM_ID',
                                                  right_on='HADM_ID')

    patient_history_df['AGE'] = (patient_history_df.INTIME.dt.year - patient_history_df.DOB.dt.year)
    patient_history_df.loc[patient_history_df.AGE < 0, 'AGE'] = 90
    # Filter out children
    # patient_history_df = patient_history_df[patient_history_df.AGE > 18]

    return patient_history_df


def make_subject_infos(patients_df,
                       admission_info_df,
                       icustays_df,
                       min_nb_stays=1,
                       max_nb_stays=1) -> pd.DataFrame:
    csv_settings = DATASET_SETTINGS["subject_info"]
    subject_info_df = merge_patient_history(patients_df, admission_info_df, icustays_df,
                                            min_nb_stays, max_nb_stays)
    subject_info_df = subject_info_df[csv_settings["columns"]]
    subject_info_df = subject_info_df.rename(columns={
        "FIRST_WARDID": "WARDID",
        "FIRST_CAREUNIT": "CAREUNIT"
    })

    return subject_info_df


def make_icu_history(patients_df,
                     admissions_df,
                     icustays_df,
                     min_nb_stays=1,
                     max_nb_stays=1) -> pd.DataFrame:
    """
    Parameters:
        patients_df:        patients data
        admissions_df:      hospital admissions data
        icustays_df:        ICU stay data

    Returns:
        icu_history_df:     Description of each ICU stay with admission data, patients data and mortality
    """
    csv_settings = DATASET_SETTINGS["icu_history"]
    icu_history_df = merge_patient_history(patients_df, admissions_df, icustays_df, min_nb_stays,
                                           max_nb_stays)

    # Inunit mortality
    mortality = icu_history_df.DOD.notnull() & ((icu_history_df.INTIME <= icu_history_df.DOD) &
                                                (icu_history_df.OUTTIME >= icu_history_df.DOD))

    mortality = mortality | (icu_history_df.DEATHTIME.notnull() &
                             ((icu_history_df.INTIME <= icu_history_df.DEATHTIME) &
                              (icu_history_df.OUTTIME >= icu_history_df.DEATHTIME)))
    icu_history_df['MORTALITY_INUNIT'] = mortality.astype(int)

    # Inhospital mortality
    mortality = icu_history_df.DOD.notnull() & ((icu_history_df.ADMITTIME <= icu_history_df.DOD) &
                                                (icu_history_df.DISCHTIME >= icu_history_df.DOD))
    mortality = mortality | (icu_history_df.DEATHTIME.notnull() &
                             ((icu_history_df.ADMITTIME <= icu_history_df.DEATHTIME) &
                              (icu_history_df.DISCHTIME >= icu_history_df.DEATHTIME)))
    icu_history_df['MORTALITY'] = mortality.astype(int)
    icu_history_df['MORTALITY_INHOSPITAL'] = mortality.astype(int)

    icu_history_df = icu_history_df[csv_settings["columns"]]
    icu_history_df = icu_history_df[icu_history_df.AGE >= 18]

    return icu_history_df


def make_diagnoses(dataset_folder, icd9codes_df, icu_history_df):
    """
    Parameters:
        dataset_folder:     If not default dataset path at data/mimic-iii-demo/
        icd9codes_df:       Diagnoses item ID defintions
        icu_history_df:     ICU stays descriptions

    Returns:
        diagnoses_df:       Dataframe containing description of diagnoses with direct link to subject and ICU stay
    """
    csv_settings = DATASET_SETTINGS["DIAGNOSES_ICD"]
    diagnoses_df = pd.read_csv(Path(dataset_folder, 'DIAGNOSES_ICD.csv'),
                               dtype=convert_dtype_dict(csv_settings["dtype"]))
    diagnoses_df = upper_case_column_names(diagnoses_df)
    diagnoses_df = diagnoses_df.merge(icd9codes_df,
                                      how='inner',
                                      left_on='ICD9_CODE',
                                      right_on='ICD9_CODE')
    diagnoses_df = diagnoses_df.merge(icu_history_df[['SUBJECT_ID', 'HADM_ID',
                                                      'ICUSTAY_ID']].drop_duplicates(),
                                      how='inner',
                                      left_on=['SUBJECT_ID', 'HADM_ID'],
                                      right_on=['SUBJECT_ID', 'HADM_ID'])

    diagnoses_df[['SUBJECT_ID', 'HADM_ID',
                  'SEQ_NUM']] = diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)

    # Clinical Classifications Software CSS defintions of phenotypes
    with Path(dataset_folder, "resources", "hcup_ccs_2015_definitions.yaml").open("r") as file:
        phenotypes_yaml = yaml.full_load(file)

    # Create definition map
    definition_map = dict()

    for phenotype in phenotypes_yaml:
        for code in phenotypes_yaml[phenotype]['codes']:
            definition_map[code] = (phenotype, phenotypes_yaml[phenotype]['use_in_benchmark'])

    diagnoses_df['HCUP_CCS_2015'] = diagnoses_df.ICD9_CODE.apply(lambda c: definition_map[c][0]
                                                                 if c in definition_map else None)
    diagnoses_df['USE_IN_BENCHMARK'] = diagnoses_df.ICD9_CODE.apply(
        lambda c: int(definition_map[c][1]) if c in definition_map else None)

    return diagnoses_df, definition_map


def make_phenotypes(diagnoses_df, definition_map):
    """
    Parameters:
        diagnoses_df:       Diagnoses descriptions with ICU stay information
        resource_folder:    If not default dataset path at data/mimic-iii-demo/resources/

    Returns:
        phenotypes_df:      Binary matrix with diagnoses over ICU stays
    """
    # Merge definitions to diagnoses
    phenotype_dictionary_df = pd.DataFrame(definition_map).T.reset_index().rename(columns={
        'index': 'ICD9_CODE',
        0: 'HCUP_CCS_2015',
        1: 'USE_IN_BENCHMARK'
    })
    phenotype_dictionary_df['use_in_benchmark'] = phenotype_dictionary_df[
        'use_in_benchmark'].astype(int)

    phenotypes_df = diagnoses_df.merge(phenotype_dictionary_df,
                                       how='inner',
                                       left_on='ICD9_CODE',
                                       right_on='ICD9_CODE')

    # Extract phenotypes from diagnoses
    phenotypes_df = phenotypes_df[['ICUSTAY_ID', 'HCUP_CCS_2015'
                                  ]][phenotypes_df.use_in_benchmark > 0].drop_duplicates()
    phenotypes_df['VALUE'] = 1

    # Definitions again icu stays
    phenotypes_df = phenotypes_df.pivot(index='icuystay_id',
                                        columns='HCUP_CCS_2015',
                                        values='VALUE')

    # Impute values and sort axes
    phenotypes_df = phenotypes_df.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)

    return phenotypes_df


def get_by_subject(df, sort_by):
    """
    Parameters:
        df:         Dataframe with subject_id columns
        sort_by:    Column by which to sort the subdataframes

    Returns:
        subject_events:     Dictionary containing events by subject_id
    """
    return {id: x for id, x in df.sort_values(by=sort_by).groupby('SUBJECT_ID')}


if __name__ == "__main__":
    # subject_groups("/home/amadou/Data/ml_data/research-internship/mimic-iii-demo")
    iterative_extraction(
        storage_path=Path("/home/amadou/CodeWorkspace/data/research-internship/processed-trials"),
        source_path=Path("/home/amadou/CodeWorkspace/data/mimic-iii-demo"),
        ehr=None,
        from_storage=False,
        chunksize=10000,
        num_subjects=None)
