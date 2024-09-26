"""
This module provides classes for processing subject events from the raw CHARTEVENT, LABEVENTS and OUTPUTEVENTS csv and store it by subject.
This class is used for multiprocessing and employed by the iterative extraction. The preprocessing is done using
the datasets.extraction.extraction_functions.extract_subject_events, while the class itself is part of the 
even processing chain.

EventProducer -> EventConsumer -> ProgressPublisher

**Subject Events**: A dictionary where each key is a subject ID, and the value is a DataFrame of 
chart events (e.g., lab results, vital signs) associated with that subject.

- From: CHARTEVENTS, LABEVENTS, OUTPUTEVENTS
- In: evenet_consumer.py
- Cols: SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE, VALUEUOM

Examples
-------- 
.. code-block:: python

    from pathlib import Path
    from multiprocess import JoinableQueue, Lock
    import pandas as pd
    from some_module import EventConsumer, EventReader, ExtractionTracker

    # Initialize parameters
    storage_path = Path('/path/to/storage')
    in_q = JoinableQueue()
    out_q = JoinableQueue()
    icu_history_df = pd.read_csv('/path/to/icu_history.csv')
    tracker = ExtractionTracker(storage_path)

    # Create and start the consumer
    consumer = EventConsumer(storage_path=storage_path,
                             in_q=in_q,
                             out_q=out_q,
                             icu_history_df=icu_history_df,
                             lock=tracker._lock)
    consumer.start()

    # Create the event reader and get a chunk of data
    event_reader = EventReader(dataset_folder='/path/to/data',
                               chunksize=100000)
    event_frames, frame_lengths = event_reader.get_chunk()
    events_df = pd.concat(event_frames.values(), ignore_index=True)

    # Put the data into the queue and join the consumer
    in_q.put((events_df, frame_lengths))
    consumer.join()
"""

import pandas as pd
from pathlib import Path
from settings import *
from utils.types import NoopLock
from utils.IO import *
from multiprocess import Process, JoinableQueue, Lock
from .extraction_functions import extract_subject_events
from ..writers import DataSetWriter


class EventConsumer(Process):
    """
    A process that consumes events from a queue, processes them, and writes the results to storage.

    Parameters
    ----------
    storage_path : Path
        Path to the storage directory where the processed data will be saved.
    in_q : JoinableQueue
        Queue from which to read the events to process.
    out_q : JoinableQueue
        Queue to which the processed events are sent.
    icu_history_df : pd.DataFrame
        DataFrame containing ICU history information.
    lock : Lock
        Lock to manage access to shared resources.

    Methods
    -------
    run()
        Runs the consumer process, processing events from the queue.
    _make_subject_events(chartevents_df, icu_history_df)
        Creates subject events from chartevents and ICU history data.
    start()
        Starts the consumer process.
    join()
        Joins the consumer process.
    """

    def __init__(self,
                 storage_path: Path,
                 in_q: JoinableQueue,
                 out_q: JoinableQueue,
                 icu_history_df: pd.DataFrame,
                 lock: Lock = NoopLock()):
        super().__init__()
        self._in_q = in_q
        self._out_q = out_q
        self._icu_history_df = icu_history_df
        self._dataset_writer = DataSetWriter(storage_path)
        self._lock = lock

    def run(self):
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
        return extract_subject_events(chartevents_df, icu_history_df)

    def start(self):
        super().start()
        debug_io("Started consumers")

    def join(self):
        super().join()
        debug_io("Joined in consumers")
