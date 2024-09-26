"""
This module provides a class for publishing the progress of the event processing chain.
This class is used for multiprocessing and employed by the iterative extraction.

EventProducer -> EventConsumer -> ProgressPublisher



Examples
--------
.. code-block:: python

    from pathlib import Path
    from multiprocess import JoinableQueue, Lock
    from ..trackers import ExtractionTracker

    # Initialize parameters
    source_path = Path('/path/to/source')
    storage_path = Path('/path/to/storage')
    in_q = JoinableQueue()
    out_q = JoinableQueue()
    icu_history_df = pd.read_csv('/path/to/icu_history.csv')
    lock = Lock()
    tracker = ExtractionTracker(storage_path)

    # Create and start the progress publisher
    progress_publisher = ProgressPublisher(n_consumers=4,
                                            source_path=source_path,
                                            in_q=out_q,
                                            tracker=tracker)
    progress_publisher.start() 

    # Create and start the event consumer
    consumer = EventConsumer(storage_path=storage_path,
                                in_q=in_q,
                                out_q=out_q,
                                icu_history_df=icu_history_df,
                                lock=lock)
    consumer.start()

    # Read and process events
    event_reader = EventReader(dataset_folder='/path/to/data', chunksize=1000)
    event_frames, frame_lengths = event_reader.get_chunk()
    events_df = pd.concat(event_frames.values(), ignore_index=True)
    in_q.put((events_df, frame_lengths))

    Processed event rows:
    CHARTEVENTS:    1000/5000
    LABEVENTS:      1000/6000
    OUTPUTEVENTS:   1000/4500
"""

import os
import multiprocessing as mp
from utils import count_csv_size
from settings import *
from utils.IO import *
from pathlib import Path
from multiprocess import JoinableQueue, Process
from ..trackers import ExtractionTracker
from utils.types import NoopLock


class ProgressPublisher(Process):
    """
    Publishes the progress of the event processing to the command line.

    Parameters
    ----------
    n_consumers : int
        Number of consumer processes.
    source_path : Path
        Path to the source directory containing the raw data files.
    in_q : JoinableQueue
        Queue from which to read the progress updates.
    tracker : ExtractionTracker
        Tracker to keep track of extraction progress.

    Methods
    -------
    run()
        Runs the progress publisher, updating and printing the progress.
    start()
        Starts the progress publisher process.
    join()
        Joins the progress publisher process.
    """

    def __init__(self,
                 n_consumers: int,
                 source_path: Path,
                 in_q: JoinableQueue,
                 tracker: ExtractionTracker,
                 verbose: bool = False,
                 lock: mp.Lock = NoopLock()):
        super().__init__()
        self._in_q = in_q
        self._lock = lock
        self._verbose = verbose
        self._tracker = tracker
        self._n_consumers = n_consumers
        self._event_file_lengths = {
            name: count_csv_size(Path(source_path, name))
            for name in ["CHARTEVENTS.csv", "LABEVENTS.csv", "OUTPUTEVENTS.csv"]
        }

    def run(self):
        done_count = 0  # Track consumer finishes

        # Print initial state
        msg = [f"Processed event rows: "]
        for csv_name in self._event_file_lengths.keys():
            with self._lock:
                csv_event_count = self._tracker.count_subject_events[csv_name]
            total_event_count = self._event_file_lengths[csv_name]
            print_name = csv_name.strip('.csv') + ": "
            msg.append(
                f"{print_name} {csv_event_count:>{len(str(total_event_count))}}/{total_event_count}"
            )
        info_io("\n".join(msg), verbose=self._verbose)

        while True:
            # Draw tracking information from queue
            frame_lengths, finished = self._in_q.get()
            # TODO! real crash resilience can only be achieved by updating in the event consumer
            with self._lock:
                self._tracker.count_subject_events += frame_lengths
            # Track consumer finishes
            if finished:
                done_count += 1
                debug_io(f"Publisher received consumer finished: {done_count}/{self._n_consumers}",
                         verbose=self._verbose)
            # Print current state
            msg = [f"Processed event rows: "]
            for csv_name in self._event_file_lengths.keys():
                with self._lock:
                    csv_event_count = self._tracker.count_subject_events[csv_name]
                total_event_count = self._event_file_lengths[csv_name]
                print_name = csv_name.strip('.csv') + ": "
                msg.append(
                    f"{print_name} {csv_event_count:>{len(str(total_event_count))}}/{total_event_count}"
                )

            info_io("\n".join(msg),
                    flush_block=not (int(os.getenv("DEBUG", 0))),
                    verbose=self._verbose)
            # Join publisher
            if done_count == self._n_consumers:
                with self._lock:
                    self._tracker.has_subject_events = True
                debug_io("All consumers finished, publisher finishes now.")
                self._in_q.task_done()
                break
            self._in_q.task_done()
        return

    def start(self):
        super().start()
        debug_io("Started publisher")

    def join(self):
        super().join()
        debug_io("Joined publisher")
