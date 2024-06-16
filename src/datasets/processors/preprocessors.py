"""
This module provides the MIMICPreprocessor class to preprocess the MIMIC-III dataset for various tasks such as in-hospital mortality, decompensation, length of stay, and phenotyping.

Usage Example
--------------
.. code-block:: python

    from pathlib import Path
    import yaml
    from datasets.readers import ExtractedSetReader
    from datasets.trackers import PreprocessingTracker
    from datasets.preprocessor import MIMICPreprocessor

    # Define the path to the dataset, storage, and phenotype configuration
    dataset_path = Path("/path/to/extracted/dataset")
    storage_path = Path("/path/to/store/processed/data")
    phenotypes_path = Path("/path/to/phenotypes.yaml")

    # Initialize the reader and tracker
    reader = ExtractedSetReader(dataset_path)
    tracker = PreprocessingTracker(storage_path)

    # Load the phenotypes configuration from a .yaml file
    with open(phenotypes_path, 'r') as file:
        phenotypes_yaml = yaml.safe_load(file)

    # Initialize the MIMICPreprocessor for the LOS (length-of-stay) task
    # Tasks are IHM, DECOMP, LOS, PHENO
    preprocessor = MIMICPreprocessor(
        phenotypes_yaml=phenotypes_yaml,
        task="LOS",
        label_type="sparse",
        reader=reader,
        tracker=tracker,
        storage_path=storage_path,
        verbose=True
    )

    # Transform the entire dataset
    reader = ExtractedSetReader(dataset_path)
    dataset = reader.read_subjects(read_ids=True)

    subject_id = 12345
    (X, y), tracking_info = preprocessor.transform_subject(subject_id)

    X, y = preprocessor.transform_dataset(dataset)

    # Alternatively transform the reader directly
    reader = preprocessor.transform_reader(reader)

    # Save the transformed data
    preprocessor.save_data()
"""
import random
import numpy as np
import dateutil
import pandas as pd
from typing import Dict, Tuple, Dict
from copy import deepcopy
from pathlib import Path
from multiprocess import Manager
from datasets.writers import DataSetWriter
from datasets.readers import ExtractedSetReader, ProcessedSetReader
from datasets.trackers import PreprocessingTracker
from datasets.processors import AbstractProcessor
from datasets.mimic_utils import copy_subject_info
from utils import dict_subset
from utils.IO import *
from settings import *


class MIMICPreprocessor(AbstractProcessor):
    """
    Preprocesses MIMIC-III dataset for various tasks such as in-hospital mortality,
    decompensation, length of stay, and phenotyping.

    Parameters
    ----------
    phenotypes_yaml : dict
        YAML configuration for phenotypes.
    task : str
        The task to perform ("IHM", "DECOMP", "LOS", "PHENO").
    label_type : str, optional
        The type of labels to use ("sparse" or "one-hot"), by default "sparse".
    reader : ExtractedSetReader, optional
        The reader to read source data, by default None.
    tracker : PreprocessingTracker, optional
        The tracker to keep track of preprocessing progress, by default None.
    storage_path : Path, optional
        The path to store processed data, by default None.
    verbose : bool, optional
        Flag for verbosity, by default False.
    """

    def __init__(self,
                 phenotypes_yaml: dict,
                 task: str,
                 label_type: str = "sparse",
                 reader: ExtractedSetReader = None,
                 tracker: PreprocessingTracker = None,
                 storage_path: Path = None,
                 verbose: bool = False):

        self._operation_name = "preprocessing"  # For printing
        self._operation_adjective = "preprocessed"
        self._save_file_type = "hdf5" if task == "MULTI" else "csv"
        if label_type not in ["sparse", "one-hot"]:
            raise ValueError(f"Type must be one of {*['sparse', 'one-hot'],}")

        self._label_type = label_type
        self._storage_path = storage_path

        if task not in TASK_NAMES:
            raise ValueError(f"Task must be one of {*TASK_NAMES,}")

        self._task = task
        self._writer = (None if storage_path is None else DataSetWriter(self._storage_path))
        self._source_reader = reader  # the set we are trying to read from
        self._lock = Manager().Lock()
        if tracker is not None:
            self._tracker = tracker
        else:
            with self._lock:
                self._tracker = (None if storage_path is None else PreprocessingTracker(
                    Path(storage_path, "progress")))
        self._processed_set_reader = (None if storage_path is None else ProcessedSetReader(
            root_path=storage_path))
        self._phenotypes_yaml = phenotypes_yaml
        self._verbose = verbose

        # Tracking variables
        self._init_tracking_variables()

        self._X = dict()
        self._y = dict()

    @property
    def subjects(self) -> list:
        """
        Get the list of subject IDs that can be processed from the reader.

        Returns
        -------
        list
            A list of subject IDs.
        """
        if self._source_reader is None:
            return []
        return self._source_reader.subject_ids

    def transform_dataset(self,
                          dataset: dict,
                          subject_ids: list = None,
                          num_subjects: int = None,
                          source_path: Path = None,
                          storage_path: Path = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Transforms and processes the dataset.

        This method processes the entire dataset, transforming the data for each subject, and then 
        saving the processed data to the specified location. It handles subject selection and ensures 
        the processed data is correctly stored.

        Parameters
        ----------
        dataset : dict
            The dataset to process.
        subject_ids : list, optional
            List of subject IDs to process. Defaults to None.
        num_subjects : int, optional
            Number of subjects to process. Defaults to None.
        source_path : Path, optional
            Source path of the data. Defaults to None.

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Processed data with keys 'X' for features and 'y' for labels.
        """
        # TODO! This also needs to work with no storage path provided
        self._init_tracking_variables()
        orig_subject_ids = deepcopy(subject_ids)
        if storage_path is not None:
            self._storage_path = storage_path
            self._tracker = PreprocessingTracker(Path(storage_path, "progress"))
        elif self._storage_path is not None and (num_subjects is not None or
                                                 subject_ids is not None):
            # Reinit to see if reprocessing is necessary,
            self._tracker.set_subject_ids(subject_ids)
            self._tracker.set_num_subjects(num_subjects)
        copy_subject_info(source_path, self._storage_path)

        if self._tracker.is_finished:
            info_io(
                f"Compact {self._operation_name} already finalized in directory:\n{str(self._storage_path)}",
                verbose=self._verbose)
            if num_subjects is not None:
                subject_ids = random.sample(self._tracker.subject_ids, k=num_subjects)
            return ProcessedSetReader(root_path=self._storage_path,
                                      subject_ids=subject_ids).read_samples(read_ids=True)

        info_io(f"Compact {self._operation_name}: {self._task}", level=0, verbose=self._verbose)

        subject_ids, exclud_subj, unknown_subj = self._get_subject_ids(
            num_subjects=num_subjects,
            subject_ids=subject_ids,
            processed_subjects=self._tracker.subject_ids,
            all_subjects=dataset.keys())
        assert all([len(subject) for subject in dataset.values()])

        if not subject_ids:
            self._tracker.is_finished = True
            info_io(f"Finalized for task {self._task} in directory:\n{str(self._storage_path)}",
                    verbose=self._verbose)
            if num_subjects and not self._n_subjects == num_subjects:
                warn_io(
                    f"The subject target was not reached, missing {self._n_subjects - num_subjects} subjects.",
                    verbose=self._verbose)
            if orig_subject_ids is not None:
                orig_subject_ids = list(set(orig_subject_ids) & set(self._tracker.subject_ids))
            return ProcessedSetReader(self._storage_path,
                                      subject_ids=orig_subject_ids).read_samples(read_ids=True)

        self._n_skip = len(unknown_subj)

        if num_subjects is not None:
            X_subjects = dict()
            y_subjects = dict()
            while not len(X_subjects) == num_subjects:
                curr_dataset = dict_subset(dataset, subject_ids)
                X, y = self._transform(dataset=curr_dataset)
                X_subjects.update(X)
                y_subjects.update(y)
                it_missing_subjects = set(X.keys()) - set(subject_ids)
                subject_ids, exclud_subj, _ = self._get_subject_ids(
                    num_subjects=num_subjects - len(self._tracker.subject_ids),  # len(X_subjects),
                    subject_ids=None,
                    processed_subjects=self._tracker.subject_ids,
                    all_subjects=exclud_subj)
                if it_missing_subjects:
                    self._n_skip += len(it_missing_subjects)
                    debug_io(f"Missing subjects are: {*it_missing_subjects,}")
                if not subject_ids:
                    break
                if len(X_subjects) == num_subjects:
                    debug_io(
                        f"Missing { len(self._tracker.subject_ids) - num_subjects} subjects."  #len(X_subjects) - num_subjects} subjects."
                    )
                    debug_io(f"Unprocessable subjects are: {*it_missing_subjects,}")

        else:
            assert all([len(subject) for subject in dataset.values()])
            dataset = dict_subset(dataset, subject_ids)
            assert all([len(subject) for subject in dataset.values()])
            (X_subjects, y_subjects) = self._transform(dataset=dataset)
        if self._storage_path is not None:
            self.save_data()
            info_io(
                f"Finalized {self._operation_name} for {self._task} in directory:\n{str(self._storage_path)}",
                verbose=self._verbose)
        else:
            info_io(f"Finalized {self._operation_name} for {self._task}.", verbose=self._verbose)
        self._tracker.is_finished = True
        if orig_subject_ids is not None:
            orig_subject_ids = list(set(orig_subject_ids) & set(self._tracker.subject_ids))
        return ProcessedSetReader(root_path=self._storage_path,
                                  subject_ids=orig_subject_ids).read_samples(read_ids=True)

    def transform_subject(self, subject_id: int) -> None:
        """
        Transforms the extracted dataset for the specified task.

        Parameters
        ----------
        dataset : dict
            The dataset to transform.

        Returns
        -------
        tuple
            A tuple containing transformed feature and label data.
        """
        subject_data = self._source_reader.read_subject(subject_id, read_ids=True)
        if not subject_data:
            return None, None
        del subject_data["subject_events"]
        X, y = self._transform({subject_id: subject_data})
        if not X or not y:
            return None, None
        if self._tracker is None:
            return X, y
        with self._lock:
            tracking_info = self._tracker.subjects[subject_id]
        return (X, y), tracking_info

    def _transform(self, dataset: dict):
        """
        Transforms the extracted dataset.

        This processes the provided extracted dataset according to the specified task, and returns the 
        processed tasks and labels along with tracking information if available.
        """
        start_verbose = True

        if self._task in ["LOS"] and self._label_type == "one-hot":
            self._label_type = "sparse"

        for subject, subject_data in dataset.items():
            skip_subject = False

            subject_timeseries: pd.DataFrame = subject_data['timeseries']
            diagnoses_df: pd.DataFrame = subject_data['subject_diagnoses']
            icuhistory_df: pd.DataFrame = subject_data['subject_icu_history']
            episodic_data_df: pd.DataFrame = subject_data['episodic_data']

            self._X[subject] = dict()
            self._y[subject] = dict()

            tracking_info = dict()
            with self._lock:
                is_in_subjects = subject in self._tracker.subjects
            if (self._tracker is not None) and \
            is_in_subjects and \
            (not self._X[subject]):
                # Do not reprocess already existing directories
                self._X[subject], self._y[subject] = self._processed_set_reader.read_sample(
                    str(subject), read_ids=True).values()
                skip_subject = True
                continue
            elif start_verbose:
                if self._verbose:
                    info_io(
                        f"Processing timeseries data:\n"
                        f"Processed subjects: {self._n_subjects}\n"
                        f"Processed stays: {self._n_stays}\n"
                        f"Processed samples: {self._n_samples}\n"
                        f"Skipped subjects: {self._n_skip}",
                        flush_block=True,
                        verbose=self._verbose)
                    start_verbose = False

            for icustay in subject_timeseries:
                stay_timeseries_df = subject_timeseries[icustay]
                cur_episodic_data_df = episodic_data_df.loc[icustay]
                cur_icuhistory_sr = icuhistory_df.reset_index().set_index("ICUSTAY_ID").loc[icustay]

                if not pd.isna(cur_episodic_data_df["MORTALITY"]):
                    mortality = int(cur_episodic_data_df["MORTALITY"])
                else:
                    continue
                if self._task == "MULTI":
                    stay_diagnoses_df = diagnoses_df[diagnoses_df['ICUSTAY_ID'] == icustay]
                    self._X[subject][icustay], self._y[subject][icustay] = self.make_multitask_data(
                        stay_timeseries_df, cur_episodic_data_df, cur_icuhistory_sr,
                        stay_diagnoses_df, self._phenotypes_yaml, mortality)
                elif self._task == "IHM":
                    self._X[subject][icustay], self._y[subject][
                        icustay] = self.make_inhospital_mortality_data(
                            stay_timeseries_df, cur_episodic_data_df, mortality)

                elif self._task == "DECOMP":
                    self._X[subject][icustay], self._y[subject][
                        icustay] = self.make_decompensation_data(stay_timeseries_df,
                                                                 cur_episodic_data_df,
                                                                 cur_icuhistory_sr)
                elif self._task == "LOS":
                    self._X[subject][icustay], self._y[subject][
                        icustay] = self.make_length_of_stay_data(stay_timeseries_df,
                                                                 cur_episodic_data_df)
                elif self._task == "PHENO":
                    stay_diagnoses_df = diagnoses_df[diagnoses_df['ICUSTAY_ID'] == icustay]
                    self._X[subject][icustay], self._y[subject][
                        icustay] = self.make_pheontyping_data(stay_timeseries_df,
                                                              cur_episodic_data_df,
                                                              stay_diagnoses_df,
                                                              self._phenotypes_yaml)
                else:
                    raise ValueError(
                        "Task must be one of: in_hospital_mortality, decompensation, length_of_stay, phenotyping"
                    )
                if self._y[subject][icustay].empty:
                    del self._y[subject][icustay]
                    del self._X[subject][icustay]
                    continue
                else:
                    tracking_info[icustay] = len(self._y[subject][icustay])
                    self._n_stays += 1
                    self._n_samples += len(self._y[subject][icustay])
                    if self._verbose:
                        info_io(
                            f"Processing timeseries data:\n"
                            f"Processed subjects: {self._n_subjects}\n"
                            f"Processed stays: {self._n_stays}\n"
                            f"Processed samples: {self._n_samples}\n"
                            f"Skipped subjects: {self._n_skip}",
                            flush_block=True,
                            verbose=self._verbose)

            if skip_subject:
                continue

            if self._tracker is not None and tracking_info:
                with self._lock:
                    self._tracker.subjects.update({subject: tracking_info})

            if not len(self._y[subject]) or not len(self._X[subject]):
                del self._y[subject]
                del self._X[subject]
                self._n_skip += 1
            else:
                self._n_subjects += 1
                # print(subject, self._n_subjects)
        if self._verbose:
            info_io(
                f"Processing timeseries data:\n"
                f"Processed subjects: {self._n_subjects}\n"
                f"Processed stays: {self._n_stays}\n"
                f"Processed samples: {self._n_samples}\n"
                f"Skipped subjects: {self._n_skip}",
                flush_block=True,
                verbose=self._verbose)

        return self._X, self._y

    def make_multitask_data(self, timeseries_df: pd.DataFrame, episodic_data_df: pd.DataFrame,
                            icu_stay: pd.DataFrame, diagnoses_df: pd.DataFrame,
                            phenotypes_yaml: pd.DataFrame, mortality: int):
        # Initialize containers for features and labels
        self._label_type = "sparse"
        mortality = int(episodic_data_df.loc["MORTALITY"])
        los = 24.0 * episodic_data_df.loc['LOS']  # in hours
        precision = MULTI_SETTINGS['sample_precision']
        sample_rate = MULTI_SETTINGS['sample_rate']

        y = {}

        # Process Phenotyping (PHENO)
        X_pheno, y_pheno = self.make_pheontyping_data(timeseries_df=timeseries_df,
                                                      episodic_data_df=episodic_data_df,
                                                      diagnoses_df=diagnoses_df,
                                                      phenotypes_yaml=phenotypes_yaml)
        if X_pheno.empty:
            return pd.DataFrame(), pd.DataFrame()
        # Process In-Hospital Mortality (IHM)
        _, y_ihm = self.make_inhospital_mortality_data(timeseries_df=timeseries_df,
                                                       episodic_data_df=episodic_data_df,
                                                       mortality=mortality)
        y['IHM_pos'] = 0 if y_ihm.empty else 47
        y['IHM_mask'] = 1 if not y_ihm.empty else 0
        y['IHM_label'] = mortality if y_ihm.empty else y_ihm['y'].iloc[0]

        # Process Decompensation (DECOMP)
        _, y_decomp = self.make_decompensation_data(timeseries_df=timeseries_df,
                                                    episodic_data_df=episodic_data_df,
                                                    icu_stay=icu_stay,
                                                    label_start_time=-1e6,
                                                    start_time=-1e6)

        sample_times = np.arange(0.0, min(los, y_decomp.index.max()) + precision, sample_rate)
        dec_start_time = DECOMP_SETTINGS['label_start_time']
        if not y_decomp.empty:
            y_decomp = y_decomp.reindex(sample_times)
            y['DECOMP_masks'] = (y_decomp.index > dec_start_time).astype(int).tolist()
            y['DECOMP_labels'] = y_decomp['y'].fillna(0).values.reshape(-1).tolist()
        else:
            sample_times = np.arange(0.0, dec_start_time + precision, sample_rate)
            y['DECOMP_masks'] = np.zeros(len(sample_times)).tolist()
            y['DECOMP_labels'] = np.zeros(len(sample_times)).tolist()

        # Process Length of Stay (LOS)
        _, y_los = self.make_length_of_stay_data(timeseries_df=timeseries_df,
                                                 episodic_data_df=episodic_data_df,
                                                 label_start_time=-1e6,
                                                 start_time=-1e6)
        los_start_time = LOS_SETTINGS['label_start_time']
        if not y_los.empty:
            sample_times = np.arange(0.0, y_los.index.max() + precision, sample_rate)
            y_los = y_los.reindex(sample_times)
            y['LOS_masks'] = (y_los.index > los_start_time).astype(int).tolist()
            y['LOS_labels'] = y_los['y'].fillna(0).values.reshape(-1).tolist()
        else:
            y['LOS_masks'] = np.zeros(len(sample_times)).tolist()
            y['LOS_labels'] = np.zeros(len(sample_times)).tolist()
        y["LOS_value"] = los

        # Pheno comes last
        if not y_pheno.empty:
            y['PHENO_labels'] = y_pheno.values.reshape(-1).tolist()
        else:
            y['PHENO_labels'] = [
                0 for phenotype, data in phenotypes_yaml.items() if data['use_in_benchmark']
            ]

        y_df = pd.DataFrame()
        for label, value in y.items():
            y_df[label] = [value]

        return X_pheno, y_df

    def make_inhospital_mortality_data(self, timeseries_df: pd.DataFrame,
                                       episodic_data_df: pd.DataFrame, mortality: int):
        """
        Prepares data for the in-hospital mortality prediction task.

        This method extracts the timeseries up to the label_start_time specified in the task settings and sets the 
        label to 1 if the patient died in the ICU afterwards.

        - Filters the timeseries data up to the label_start_time.
        - Sets the label to 1 if the patient died in the ICU after the label_start_time.
        - Returns the filtered timeseries data and corresponding labels.
        """
        precision = IHM_SETTINGS['sample_precision']
        label_start_time = IHM_SETTINGS['label_start_time']
        los = 24.0 * episodic_data_df.loc['LOS']  # in hours
        if los < label_start_time:
            return pd.DataFrame(columns=timeseries_df.columns), \
                   pd.DataFrame(columns=['Timestamp', 'y']).set_index('Timestamp')
        X: pd.DataFrame = timeseries_df[(timeseries_df.index < label_start_time + precision)
                                        & (timeseries_df.index > -precision)]
        if X.empty:
            return X, pd.DataFrame(columns=['Timestamp', 'y'])

        y = np.array([(X.index[-1], mortality)])
        y = pd.DataFrame(y, columns=['Timestamp', 'y']).set_index('Timestamp')

        return X, y

    def make_decompensation_data(self,
                                 timeseries_df: pd.DataFrame,
                                 episodic_data_df: pd.DataFrame,
                                 icu_stay,
                                 label_start_time: float = None,
                                 start_time: float = None):
        """
        Prepares data for the decompensation prediction task.

        This method extracts relevant features from timeseries data starting at ICU admission time, with the first 
        label generated at the label_start_time. It checks at every sample_rate interval if the patient died within 
        the future_time_interval (sets label to 1 if true).

        - Filters the timeseries data up to the LOS duration.
        - Generates sample times at specified intervals starting from the label_start_time.
        - Checks if the patient died within the future_time_interval at each sample time.
        - Returns the filtered timeseries data and corresponding labels.
        """
        precision = DECOMP_SETTINGS['sample_precision']
        sample_rate = DECOMP_SETTINGS['sample_rate']
        label_start_time = DECOMP_SETTINGS[
            'label_start_time'] if label_start_time is None else label_start_time
        future_time_interval = DECOMP_SETTINGS['future_time_interval']
        # future_time_interval is Hours in which the preson will have to be dead for label to be 1

        mortality = int(episodic_data_df.loc["MORTALITY"])

        los = 24.0 * episodic_data_df.loc['LOS']  # in hours

        deathtime = icu_stay['DEATHTIME']
        intime = icu_stay['INTIME']

        if pd.isnull(deathtime):
            lived_time = 1e18
        else:
            if isinstance(icu_stay['DEATHTIME'], str):
                deathtime = dateutil.parser.parse(deathtime)
            if isinstance(icu_stay['INTIME'], str):
                intime = dateutil.parser.parse(intime)
            lived_time = (deathtime - intime).total_seconds() / 3600.0

        X: pd.DataFrame = timeseries_df[(timeseries_df.index < los + precision)
                                        & (timeseries_df.index > -precision)]
        if X.empty:
            return X, pd.DataFrame(columns=['Timestamp', 'y']).set_index('Timestamp')

        event_times = timeseries_df.index[(timeseries_df.index < los + precision)
                                          & (timeseries_df.index > -precision)]

        sample_times = np.arange(0.0, min(los, lived_time) + precision, sample_rate)
        sample_times = list(filter(lambda x: x > label_start_time, sample_times))

        # At least one measurement
        sample_times = list(
            filter(lambda x: x > event_times[0]
                   if start_time is None else start_time, sample_times))

        y = list()

        for hour in sample_times:
            if mortality == 0:
                cur_mortality = 0
            else:
                cur_mortality = int(lived_time - hour < future_time_interval)
            y.append((hour, cur_mortality))

        if not y:
            return pd.DataFrame(columns=X.columns), pd.DataFrame(columns=['Timestamp', 'y'])

        y = pd.DataFrame(y, columns=['Timestamp', 'y']).set_index('Timestamp')

        return X, y

    def make_length_of_stay_data(self,
                                 timeseries_df: pd.DataFrame,
                                 episodic_data_df: pd.DataFrame,
                                 label_start_time: int = None,
                                 start_time: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares data for the length of stay prediction task.

        This method extracts relevant features from the timeseries data starting at ICU admission time and generates 
        labels for the remaining length of stay (LOS). The first label is generated at the label_start_time, and 
        subsequent labels are generated at intervals specified by the sample_rate. The LOS is divided into categories 
        using specified bins, and labels are created either in "one-hot" format (categorical) or "sparse" format 
        (continuous) based on the remaining LOS.

        - Filters the timeseries data within the LOS duration.
        - Generates sample times at specified intervals.
        - Creates labels in either "one-hot" or "sparse" format based on the remaining LOS.
        - Returns the filtered timeseries data and corresponding labels.
        """
        precision = LOS_SETTINGS['sample_precision']
        sample_rate = LOS_SETTINGS['sample_rate']
        label_start_time = LOS_SETTINGS[
            'label_start_time'] if label_start_time is None else label_start_time
        bins = LOS_SETTINGS['bins']

        los = 24.0 * episodic_data_df.loc['LOS']  # in hours

        X: pd.DataFrame = timeseries_df[(timeseries_df.index < los + precision)
                                        & (timeseries_df.index > -precision)]
        if X.empty:
            return X, pd.DataFrame(columns=['Timestamp', 'y']).set_index('Timestamp')

        event_times = timeseries_df.index[(timeseries_df.index < los + precision)
                                          & (timeseries_df.index > -precision)]

        sample_times = np.arange(0.0, los + precision, sample_rate)
        sample_times = list(filter(lambda x: x > label_start_time, sample_times))
        sample_times = list(
            filter(lambda x: x >= event_times[0]
                   if start_time is None else start_time, sample_times))

        y = list()

        for hour in sample_times:
            if self._label_type == "one-hot":
                category_index = np.digitize(los - hour, bins) - 1
                remaining_los = [0] * 10
                remaining_los[category_index] = 1
                y.append((hour, remaining_los))

            elif self._label_type == "sparse":
                y.append((hour, los - hour))

        y = pd.DataFrame(y, columns=['Timestamp', 'y']).set_index('Timestamp')
        if y.empty:
            # Will prevent the sample from being used
            return pd.DataFrame(columns=X.columns), y

        return X, y

    def make_pheontyping_data(self, timeseries_df: pd.DataFrame, episodic_data_df: pd.DataFrame,
                              diagnoses_df: pd.DataFrame, phenotypes_yaml: pd.DataFrame):
        """
        Prepares data for the phenotyping task.

        This method extracts relevant features from the timeseries data starting at ICU admission time and creates 
        labels for phenotypes based on the provided YAML configuration. It maps diagnosis codes to phenotype groups, 
        filters the labels to include only those used in the benchmark, and returns the filtered timeseries data and 
        corresponding phenotype labels.

        - Filters the timeseries data within the length of stay (LOS) duration.
        - Maps diagnosis codes to phenotype groups using the YAML configuration.
        - Creates labels for phenotypes, including only those used in the benchmark.
        - Returns the filtered timeseries data and phenotype labels.
        """
        precision = PHENOT_SETTINGS['sample_precision']
        valid_ids = PHENOT_SETTINGS['valid_ids']

        los = 24.0 * episodic_data_df.loc['LOS']  # in hours
        X: pd.DataFrame = timeseries_df[(timeseries_df.index < los + precision)
                                        & (timeseries_df.index > -precision)]
        if X.empty:
            return X, pd.DataFrame(columns=['Timestamp', 'y']).set_index('Timestamp')

        # Dictionary mapping codes to groups
        code_to_group = {}
        for group in phenotypes_yaml:
            codes = phenotypes_yaml[group]['codes']
            for code in codes:
                if code not in code_to_group:
                    code_to_group[code] = group
                else:
                    assert code_to_group[code] == group

        # index is ID of phenotype in the yaml
        id_to_group = sorted(phenotypes_yaml.keys())
        group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

        # Index is pheontype designation and item is id
        cur_labels = [0] * len(id_to_group)

        for index, row in diagnoses_df.iterrows():
            if row['USE_IN_BENCHMARK']:
                code = row['ICD9_CODE']
                group = code_to_group[code]
                group_id = group_to_id[group]
                cur_labels[group_id] = 1

        # Only use in benchmark = True labels
        if not valid_ids:
            valid_ids = [
                i for (i, x) in enumerate(cur_labels)
                if phenotypes_yaml[id_to_group[i]]['use_in_benchmark']
            ]
            PHENOT_SETTINGS['valid_ids'] = valid_ids

        cur_labels = [cur_labels[index] for index in valid_ids]

        y = [los] + cur_labels
        y = pd.DataFrame((y,), columns=["Timestamp"] + [f"y{i}" for i in range(len(y) - 1)])
        y = y.set_index('Timestamp')

        return X, y
