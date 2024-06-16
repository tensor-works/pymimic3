"""
Dataset Extraction Module
=========================

This module provides the EventProducer class for reading and processing subject events from the raw CHARTEVENT, LABEVENTS and OUTPUTEVENTS CSVs and 
handling their processing into subject events. This class is the head of the event processing chain, spawning event consumers and the progress publisher.

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
    import pandas as pd
    from some_module import EventProducer, ExtractionTracker

    # Initialize parameters
    source_path = Path('/path/to/source')
    storage_path = Path('/path/to/storage')
    chunksize = 100000
    icu_history_df = pd.read_csv('/path/to/icu_history.csv')
    tracker = ExtractionTracker(storage_path)
    subject_ids = [1234, 1235, 1236, 1237, 1238]

    # Create and run the producer
    producer = EventProducer(source_path, storage_path, chunksize, tracker, icu_history_df, subject_ids)
    producer.run()
"""

import pandas as pd
import pathos, multiprocess
from pathlib import Path
from copy import deepcopy
from multiprocess import Manager
from multiprocessing import Lock
from pathos.multiprocessing import cpu_count
from utils.IO import *
from .event_consumer import EventConsumer
from .progress_publisher import ProgressPublisher
from ..mimic_utils import *
from ..trackers import ExtractionTracker
from ..readers import EventReader


class EventProducer(object):
    """
    Produces events by reading data from the source, processing it, and passing it to consumers.
    The class handles the processing chain, that is the producer spawn event consumers and the progress publisher.
    It reads the events from the LABOUTPUTS, CHARTEVENTS and OUTPUTEVENTS CSV's and passes them via
    queues to the event consumer, which passes its progress to the progress publisher. Once dones,
    the producer joins all other processes.     

    Parameters
    ----------
    source_path : Path
        Path to the source directory containing the raw data files.
    storage_path : Path
        Path to the storage directory where the processed data will be saved.
    num_samples : int
        Number of samples to process.
    chunksize : int
        Size of chunks to read at a time.
    tracker : ExtractionTracker
        Tracker to keep track of extraction progress.
    icu_history_df : pd.DataFrame
        DataFrame containing ICU history information.
    subject_ids : list of int, optional
        List of subject IDs to process. If None, all subjects are processed. Default is None.

    Methods
    -------
    run()
        Starts the event production process, reading data, processing it, and passing it to consumers.
    _count_timeseries_sample(varmap_df, frame)
        Counts the number of unique time series samples in a DataFrame.
    _update_total_lengths(ts_lengths)
        Updates the total lengths of time series samples.
    """

    def __init__(self,
                 source_path: Path,
                 storage_path: Path,
                 num_samples: int,
                 chunksize: int,
                 tracker: ExtractionTracker,
                 icu_history_df: pd.DataFrame,
                 subject_ids: list = None,
                 verbose: bool = False):
        super().__init__()
        self._verbose = verbose
        self._source_path = source_path
        self._storage_path = storage_path
        self._tracker = tracker
        self._icu_history_df = icu_history_df
        self._num_samples = num_samples
        self._chunksize = chunksize
        self._subject_ids = subject_ids

        # Counting variables
        self._count = 0  # count read data chunks
        self._lock = Lock()
        with self._lock:
            self._total_length = self._tracker.count_total_samples
        event_csv = ["CHARTEVENTS.csv", "LABEVENTS.csv", "OUTPUTEVENTS.csv"]
        self._ts_total_lengths = dict(zip(event_csv, [0] * 3))

        # Queues to connect process stages
        manager = Manager()
        self._in_q = manager.Queue()
        self._out_q = manager.Queue()

        # Number of cpus is adjusted depending on sample size
        if num_samples is not None:
            self._cpus = min(
                cpu_count() - 2,
                int(np.ceil((num_samples - tracker.count_total_samples) / (chunksize))))
        else:
            self._cpus = cpu_count() - 2
        debug_io(f"Using {self._cpus+2} CPUs!")

    def run(self):
        # Start event consumers (processing and storing)
        consumers = [
            EventConsumer(storage_path=self._storage_path,
                          in_q=self._in_q,
                          out_q=self._out_q,
                          icu_history_df=self._icu_history_df,
                          lock=self._lock)
            # lock=multiprocess.Lock())
        ]

        consumers[0].start()
        # Starting publisher (publish processing progress)
        progress_publisher = ProgressPublisher(n_consumers=self._cpus,
                                               source_path=self._source_path,
                                               in_q=self._out_q,
                                               tracker=self._tracker,
                                               verbose=self._verbose,
                                               lock=self._lock)
        progress_publisher.start()
        event_reader = EventReader(dataset_folder=self._source_path,
                                   chunksize=self._chunksize,
                                   subject_ids=self._subject_ids,
                                   tracker=self._tracker,
                                   verbose=self._verbose,
                                   lock=self._lock)
        while True:
            # Create more consumers if needed
            if len(consumers) < self._cpus and not self._in_q.empty():
                consumers.append(
                    EventConsumer(storage_path=self._storage_path,
                                  in_q=self._in_q,
                                  out_q=self._out_q,
                                  icu_history_df=self._icu_history_df,
                                  lock=self._lock))
                consumers[-1].start()
            # Read data chunk from CSVs through readers
            event_frames, frame_lengths = event_reader.get_chunk()

            # Number of timeseries samples after preprocessing
            # TODO! This shouldn't be computed if num_samples is unspecified
            ts_lengths = {
                csv_name: self._count_timeseries_sample(event_reader._varmap_df, frame)
                for csv_name, frame in event_frames.items()
            }

            # If sample limit defined
            if self._num_samples is not None:
                # If remaining samples until sample limit is reached is smaller than sum of remaining samples
                if self._num_samples - self._total_length < sum(ts_lengths.values()):
                    # Get the correct number of samples
                    event_frames, frame_lengths, ts_lengths = get_samples_per_df(
                        event_frames, self._num_samples - self._total_length)
                    # Let consumers know about sample limit finish, so that read chunks is not incremented
                    frame_lengths["sample_limit"] = True
                    # Queue up data frame
                    events_df = pd.concat(event_frames.values(), ignore_index=True)
                    self._in_q.put((events_df, frame_lengths))

                    # Close consumers on empty dfs
                    for _ in range(self._cpus):
                        self._in_q.put((None, {}))
                    self._update_total_lengths(ts_lengths)

                    # Record total length
                    self._total_length += sum(ts_lengths.values())
                    with self._lock:
                        self._tracker.count_total_samples = self._total_length
                    self._count += 1
                    debug_io(
                        f"Event producer finished on sample size restriction and produced {self._count} event chunks."
                    )
                    break

            # Queue up data frame
            events_df = pd.concat(event_frames.values(), ignore_index=True)

            # Close consumers on empty df
            if event_reader.done_reading:
                for _ in range(len(consumers)):
                    self._in_q.put((None, {}))
                debug_io(
                    f"Event producer finished on empty data frame and produced {self._count} event chunks."
                )
                break
            else:
                self._in_q.put((events_df, frame_lengths))

            # Update tracking
            self._count += 1
            self._total_length += sum(ts_lengths.values())
            self._update_total_lengths(ts_lengths)
            with self._lock:
                self._tracker.count_total_samples = self._total_length

        # Join processes and queues when done reading
        debug_io(f"Joining in queue")
        self._in_q.join()
        debug_io(f"In queue joined")
        [consumer.join() for consumer in consumers]
        # signal the publisher that we're done
        debug_io(f"Sending consumer finishes for uninitialized consumers: {frame_lengths}")
        [self._out_q.put(({}, True)) for _ in range(self._cpus - len(consumers))]
        debug_io(f"Joining out queue")
        self._out_q.join()
        debug_io(f"Out queue joined")
        progress_publisher.join()

        return

    def _count_timeseries_sample(self, varmap_df: pd.DataFrame, frame: pd.DataFrame):
        frame = deepcopy(frame)
        if frame.empty:
            return 0
        frame = frame.merge(varmap_df, left_on='ITEMID', right_index=True)
        frame = frame.dropna(subset=['HADM_ID'])
        frame = frame[['HADM_ID', 'ICUSTAY_ID',
                       'CHARTTIME']].merge(self._icu_history_df[['HADM_ID', 'ICUSTAY_ID']],
                                           left_on=['HADM_ID'],
                                           right_on=['HADM_ID'],
                                           suffixes=['', '_r'],
                                           how='inner')
        frame['ICUSTAY_ID'] = frame['ICUSTAY_ID'].fillna(frame['ICUSTAY_ID_r'])
        frame = frame.dropna(subset=['ICUSTAY_ID'])
        frame = frame[frame['ICUSTAY_ID'] == frame['ICUSTAY_ID_r']]

        return int(frame[["CHARTTIME"]].nunique().values.squeeze())

    def _update_total_lengths(self, ts_lengths: dict):
        for csv_name, value in ts_lengths.items():
            self._ts_total_lengths[csv_name] += value
        return
