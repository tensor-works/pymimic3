import pandas as pd
from pathlib import Path
from utils.IO import *
from multiprocess import Process, JoinableQueue, Lock
from .extraction_functions import make_subject_events
from ..writers import DataSetWriter


class EventConsumer(Process):

    def __init__(self, storage_path: Path, in_q: JoinableQueue, out_q: JoinableQueue,
                 icu_history_df: pd.DataFrame, lock: Lock):
        """_summary_

        Args:
            in_q (JoinableQueue): _description_
            out_q (JoinableQueue): _description_
            icu_history_df (pd.DataFrame): _description_
            storage_path (Path): _description_
            lock (Lock): _description_
            subject_ids (list, optional): _description_. Defaults to None.
        """
        super().__init__()
        self._in_q = in_q
        self._out_q = out_q
        self._icu_history_df = icu_history_df
        self._dataset_writer = DataSetWriter(storage_path)
        self._lock = lock

    def run(self):
        """_summary_
        """
        count = 0  # count read data chunks
        while True:
            # Draw from queue
            chartevents_df, frame_lengths = self._in_q.get()

            if chartevents_df is None:
                # Join the consumer and update the queue
                self._in_q.task_done()
                self._out_q.put((frame_lengths, True))
                debug_io(f"Consumer finished on empty df and consumed {count} event chunks.")
                break
            # Process events
            subject_events = self._make_subject_events(chartevents_df, self._icu_history_df)
            self._dataset_writer.write_subject_events(subject_events, self._lock)

            # Update queue and send tracking data to publisher
            self._in_q.task_done()
            self._out_q.put((frame_lengths, False))
            # Update tracking
            count += 1
        return

    @staticmethod
    def _make_subject_events(chartevents_df: pd.DataFrame, icu_history_df: pd.DataFrame):
        """_summary_

        Args:
            chartevents_df (pd.DataFrame): _description_
            icu_history_df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        return make_subject_events(chartevents_df, icu_history_df)

    def start(self):
        """_summary_
        """
        super().start()
        debug_io("Started consumers")

    def join(self):
        """_summary_
        """
        super().join()
        debug_io("Joined in consumers")
