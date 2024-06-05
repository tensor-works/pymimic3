"""
This module provides the TimeseriesProcessor class for processing time series data from ICU records. 
It reads and processes subject event data, diagnoses, and ICU history to generate time series and episodic data.
The timeseries and episodic data are stored in the subject ID labeled directories, while the episodic info is 
condensed into a single CSV file.


Classes
-------
- TimeseriesProcessor(storage_path, source_path, tracker, subject_ids, diagnoses_df, icu_history_df, varmap_df, num_samples)
    Processes time series data for given subjects and stores the results.

Examples
--------
.. code-block:: python

    from pathlib import Path
    from ..trackers import ExtractionTracker
    from datasets.extraction import get_by_subject
    import pandas as pd

    # Initialize parameters
    storage_path = Path('/path/to/storage')
    source_path = Path('/path/to/source')
    tracker = ExtractionTracker(storage_path)
    subject_ids = [123, 456, 789]
    diagnoses_df = pd.read_csv('/path/to/diagnoses.csv')
    icu_history_df = pd.read_csv('/path/to/icu_history.csv')
    varmap_df = pd.read_csv('/path/to/varmap.csv')
    num_samples = 100

    # Prepare subject-specific data
    subject_diagnoses = get_by_subject(diagnoses_df)
    subject_icu_history = get_by_subject(icu_history_df)

    # Create and run the processor
    processor = TimeseriesProcessor(storage_path, source_path, tracker, subject_ids, diagnoses_df, icu_history_df, varmap_df, num_samples)
    processor.run()
"""

import pandas as pd
from utils.IO import *
from settings import *
from pathlib import Path
from pathos.multiprocessing import cpu_count, Pool
from .extraction_functions import extract_timeseries
from ..trackers import ExtractionTracker
from ..writers import DataSetWriter
from ..readers import ExtractedSetReader


class TimeseriesProcessor(object):

    def __init__(self,
                 storage_path: Path,
                 source_path: Path,
                 tracker: ExtractionTracker,
                 subject_ids: list,
                 diagnoses_df: pd.DataFrame,
                 icu_history_df: pd.DataFrame,
                 varmap_df: pd.DataFrame,
                 num_samples: int = None):
        """
        Processes time series data for given subjects and stores the results.

        Parameters
        ----------
        storage_path : Path
            Path to the storage directory where the processed data will be saved.
        source_path : Path
            Path to the source directory containing the raw data files.
        tracker : ExtractionTracker
            Tracker to keep track of extraction progress.
        subject_ids : list of int
            List of subject IDs to process.
        diagnoses_df : pd.DataFrame
            DataFrame containing diagnoses information.
        icu_history_df : pd.DataFrame
            DataFrame containing ICU history information.
        varmap_df : pd.DataFrame
            DataFrame containing variable mappings.
        num_samples : int, optional
            Number of samples to process. Default is None.
        """
        self._storage_path = storage_path
        self._tracker = tracker
        self._dataset_reader = ExtractedSetReader(source_path)
        self._dataset_writer = DataSetWriter(storage_path)
        if subject_ids is None:
            self._subject_ids = [
                int(folder.name)
                for folder in storage_path.iterdir()
                if folder.is_dir() and folder.name.isnumeric()
            ]
            self._subject_ids.sort()
        else:
            self._subject_ids = subject_ids
        self._num_samples = num_samples
        self._subject_diagnoses = diagnoses_df
        self._subject_icu_history = icu_history_df
        self._varmap_df = varmap_df

    def _store_df_chunk(self, episodic_info_df):
        """Stores a chunk of episodic information DataFrame to a CSV file.
        """
        file_path = Path(self._storage_path, "episodic_info_df.csv")
        if not file_path.is_file():
            episodic_info_df.to_csv(file_path, index=False)
        else:
            episodic_info_df.to_csv(file_path, mode='a', header=False, index=False)

    @staticmethod
    def _process_subject(subject_id):
        """Processes data for a single subject to generate episodic and time series data.
        """
        subject_event_df_path = Path(storage_path_pr, str(subject_id), "subject_events.csv")
        subject_event_df = dataset_reader_pr.read_csv(
            subject_event_df_path, dtypes=DATASET_SETTINGS["subject_events"]["dtype"])

        if subject_ids_pr is not None:
            subject_event_df = subject_event_df[subject_event_df["SUBJECT_ID"].isin(subject_ids_pr)]

        if subject_event_df.empty or \
           not subject_id in subject_diagnoses_pr or \
           not subject_id in subject_icu_history_pr:
            return pd.DataFrame(columns=["SUBJECT_ID", "ICUSTAY_ID", "Height", "Weight"])

        curr_subject_event = {subject_id: subject_event_df}
        curr_subject_diagnoses = {subject_id: subject_diagnoses_pr[subject_id]}
        curr_icu_history_pr = {subject_id: subject_icu_history_pr[subject_id]}

        episodic_data, timeseries = extract_timeseries(curr_subject_event, curr_subject_diagnoses,
                                                       curr_icu_history_pr, varmap_df_pr)

        # Store processed subject events
        name_data_pairs = {
            "episodic_data": episodic_data,
            "timeseries": timeseries,
        }
        dataset_writer_pr.write_bysubject(name_data_pairs, exists_ok=True)

        info_dfs = list()
        # Return episodic information for compact storage as it is still needed for subsequent processing
        for subject_id, df in episodic_data.items():
            df["SUBJECT_ID"] = subject_id
            df = df.reset_index()
            df = df.rename(columns={"Icustay": "ICUSTAY_ID"})
            info_dfs.append(df[["SUBJECT_ID", "ICUSTAY_ID", "Height", "Weight"]])

        if not info_dfs:
            return pd.DataFrame(columns=["SUBJECT_ID", "ICUSTAY_ID", "Height", "Weight"])

        return pd.concat(info_dfs)

    @staticmethod
    def _init(storage_path: Path, subject_ids: list, diagnoses: dict, icu_history: dict,
              varmap: pd.DataFrame, dataset_reader: ExtractedSetReader,
              dataset_writer: DataSetWriter):
        """Initializes global variables for multiprocessing pool.
        """
        global storage_path_pr
        global subject_ids_pr
        global subject_diagnoses_pr
        global subject_icu_history_pr
        global varmap_df_pr
        global dataset_writer_pr
        global dataset_reader_pr
        storage_path_pr = storage_path
        subject_ids_pr = subject_ids
        subject_diagnoses_pr = diagnoses
        subject_icu_history_pr = icu_history
        varmap_df_pr = varmap
        dataset_reader_pr = dataset_reader
        dataset_writer_pr = dataset_writer

    def run(self):
        """Runs the time series processing using multiprocessing.

        This method initializes a multiprocessing pool to process subjects in parallel,
        generating episodic and time series data for each subject and storing the results.

        Examples
        --------
        >>> processor = TimeseriesProcessor(storage_path, source_path, tracker, subject_ids, diagnoses_df, icu_history_df, varmap_df, num_samples)
        >>> processor.run()
        """
        with Pool(cpu_count() - 1,
                  initializer=self._init,
                  initargs=(self._storage_path, self._subject_ids, self._subject_diagnoses,
                            self._subject_icu_history, self._varmap_df, self._dataset_reader,
                            self._dataset_writer)) as pool:
            res = pool.imap_unordered(self._process_subject, self._subject_ids, chunksize=500)

            episodic_info_df = None
            for index, info_df in enumerate(res):
                if episodic_info_df is None:
                    episodic_info_df = info_df
                else:
                    episodic_info_df = pd.concat([episodic_info_df, info_df])

                self._tracker.subject_ids.extend(info_df["SUBJECT_ID"].unique())
                info_io(f"Subject directories extracted: {len(self._tracker.subject_ids)}",
                        end="\r",
                        flush=True)

                if index % 100 == 0 and index != 0:
                    self._store_df_chunk(episodic_info_df)
                    episodic_info_df = None

            if episodic_info_df is not None:
                self._store_df_chunk(episodic_info_df)

            self._tracker.has_episodic_data = True
            self._tracker.has_timeseries = True
        return
