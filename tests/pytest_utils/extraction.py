import pandas as pd
from typing import Dict
from utils.IO import *
from tests.tsettings import *
from tests.pytest_utils.general import assert_dataframe_equals


def compare_extracted_datasets(generated_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
                               test_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]]):
    # Filtypes are: timeseries, subject_events, episodic_data, subject_icu_stays
    tests_io(f"Comparing extracted datasets\n"
             f"Compared subjects: {0}\n"
             f"Compared timeseries: {0}\n"
             f"Compared subject events: {0}\n"
             f"Compared episodic data: {0}"
             f"Compared stay data: {0}")
    n_subjects = 0
    n_timeseries = 0
    n_subject_events = 0
    n_episodic_data = 0
    n_stay_data = 0
    for subject_id, subject_data in generated_data.items():
        for file_type, type_data in subject_data.items():
            if file_type == "timeseries":
                for stay_id, stay_data in type_data.items():
                    # Timeseries is structured
                    assert_dataframe_equals(stay_data, test_data[subject_id][file_type][stay_id])
                    n_timeseries += 1
            elif file_type == "subject_events":
                sorted_type_data = type_data.sort_values(
                    by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID"])
                sorted_type_data = sorted_type_data.reset_index(drop=True)
                sorted_gt_data = test_data[subject_id][file_type].sort_values(
                    by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID"])
                sorted_gt_data = sorted_gt_data.reset_index(drop=True)
                assert_dataframe_equals(sorted_type_data, sorted_gt_data)
                n_subject_events += 1
            else:
                assert_dataframe_equals(type_data, test_data[subject_id][file_type])
                if file_type == "episodic_data":
                    n_episodic_data += 1
                else:
                    n_stay_data += 1

            tests_io(
                f"Comparing extracted datasets\n"
                f"Compared subjects: {n_subjects}\n"
                f"Compared timeseries: {n_timeseries}\n"
                f"Compared subject events: {n_subject_events}\n"
                f"Compared episodic data: {n_episodic_data}\n"
                f"Compared stay data: {n_stay_data}",
                flush_block=True)

        n_subjects += 1

    tests_io(
        f"Comparing extracted datasets\n"
        f"Compared subjects: {n_subjects}\n"
        f"Compared timeseries: {n_timeseries}\n"
        f"Compared subject events: {n_subject_events}\n"
        f"Compared episodic data: {n_episodic_data}\n"
        f"Compared stay data: {n_stay_data}",
        flush_block=True)
