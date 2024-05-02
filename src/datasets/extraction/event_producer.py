import pandas as pd
import pathos, multiprocess
from pathlib import Path
from copy import deepcopy
from multiprocess import Manager
from pathos.multiprocessing import cpu_count
from utils.IO import *
from .event_consumer import EventConsumer
from .progress_publisher import ProgressPublisher
from ..mimic_utils import *
from ..trackers import ExtractionTracker
from ..readers import EventReader


class EventProducer(object):

    def __init__(self,
                 source_path: Path,
                 storage_path: Path,
                 num_samples: int,
                 chunksize: int,
                 tracker: ExtractionTracker,
                 icu_history_df: pd.DataFrame,
                 subject_ids: list = None):
        """_summary_

        Args:
            source_path (Path): _description_
            storage_path (Path): _description_
            num_samples (int): _description_
            chunksize (int): _description_
            tracker (MIMICExtractionTracker): _description_
            icu_history_df (pd.DataFrame): _description_
            subject_ids (list, optional): _description_. Defaults to None.
        """
        super().__init__()
        self._source_path = source_path
        self._storage_path = storage_path
        self._tracker = tracker
        self._icu_history_df = icu_history_df
        self._num_samples = num_samples
        self._chunksize = chunksize
        self._subject_ids = subject_ids

        # Counting variables
        self._count = 0  # count read data chunks
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
        """_summary_
        """
        # Start event consumers (processing and storing)
        consumers = [
            EventConsumer(storage_path=self._storage_path,
                          in_q=self._in_q,
                          out_q=self._out_q,
                          icu_history_df=self._icu_history_df,
                          lock=self._tracker._lock)
                          # lock=multiprocess.Lock())
        ]

        consumers[0].start()
        # Starting publisher (publish processing progress)
        progress_publisher = ProgressPublisher(n_consumers=self._cpus,
                                               source_path=self._source_path,
                                               in_q=self._out_q,
                                               tracker=self._tracker)
        progress_publisher.start()
        event_reader = EventReader(dataset_folder=self._source_path,
                                   chunksize=self._chunksize,
                                   subject_ids=self._subject_ids,
                                   tracker=self._tracker)
        while True:
            # Create more consumers if needed
            if len(consumers) < self._cpus and not self._in_q.empty():
                consumers.append(
                    EventConsumer(storage_path=self._storage_path,
                                  in_q=self._in_q,
                                  out_q=self._out_q,
                                  icu_history_df=self._icu_history_df,
                                  lock=self._tracker._lock))
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
        """_summary_

        Args:
            varmap_df (pd.DataFrame): _description_
            frame (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
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

        return int(frame[["CHARTTIME"]].nunique())

    def _update_total_lengths(self, ts_lengths: dict):
        """_summary_

        Args:
            ts_lengths (dict): _description_
        """
        for csv_name, value in ts_lengths.items():
            self._ts_total_lengths[csv_name] += value
        return
