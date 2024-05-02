import pandas as pd
from utils.IO import *
from settings import *
from pathlib import Path
from pathos.multiprocessing import cpu_count, Pool
from .extraction_functions import make_timeseries
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
        """_summary_

        Args:
            storage_path (Path): _description_
            source_path (Path): _description_
            tracker (ExtractionTracker): _description_
            subject_ids (list): _description_
            diagnoses_df (pd.DataFrame): _description_
            icu_history_df (pd.DataFrame): _description_
            varmap_df (pd.DataFrame): _description_
            num_samples (int, optional): _description_. Defaults to None.
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
        """_summary_

        Args:
            episodic_info_df (_type_): _description_
        """
        file_path = Path(self._storage_path, "episodic_info_df.csv")
        if not file_path.is_file():
            episodic_info_df.to_csv(file_path, index=False)
        else:
            episodic_info_df.to_csv(file_path, mode='a', header=False, index=False)

    @staticmethod
    def _process_subject(subject_id):
        """_summary_

        Args:
            subject_id (_type_): _description_

        Returns:
            _type_: _description_
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

        episodic_data, timeseries = make_timeseries(curr_subject_event, curr_subject_diagnoses,
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
        """_summary_

        Args:
            storage_path (Path): _description_
            subject_ids (list): _description_
            diagnoses (dict): _description_
            icu_history (dict): _description_
            varmap (pd.DataFrame): _description_
            dataset_reader (ExtractedSetReader): _description_
            dataset_writer (DataSetWriter): _description_
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
        """_summary_
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
