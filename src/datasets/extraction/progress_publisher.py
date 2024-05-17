import os
from utils import count_csv_size
from utils.IO import *
from pathlib import Path
from multiprocess import JoinableQueue, Process
from ..trackers import ExtractionTracker


class ProgressPublisher(Process):

    def __init__(self, n_consumers: int, source_path: Path, in_q: JoinableQueue,
                 tracker: ExtractionTracker):
        super().__init__()
        self._in_q = in_q
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
            csv_event_count = self._tracker.count_subject_events[csv_name]
            total_event_count = self._event_file_lengths[csv_name]
            print_name = csv_name.strip('.csv') + ": "
            msg.append(
                f"{print_name} {csv_event_count:>{len(str(total_event_count))}}/{total_event_count}"
            )
        info_io("\n".join(msg))

        while True:
            # Draw tracking information from queue
            frame_lengths, finished = self._in_q.get()
            # TODO! real crash resilience can only be achieved by updating in the event consumer
            self._tracker.count_subject_events += frame_lengths
            # Track consumer finishes
            if finished:
                done_count += 1
                debug_io(f"Publisher received consumer finished: {done_count}/{self._n_consumers}")
            # Print current state
            msg = [f"Processed event rows: "]
            for csv_name in self._event_file_lengths.keys():
                csv_event_count = self._tracker.count_subject_events[csv_name]
                total_event_count = self._event_file_lengths[csv_name]
                print_name = csv_name.strip('.csv') + ": "
                msg.append(
                    f"{print_name} {csv_event_count:>{len(str(total_event_count))}}/{total_event_count}"
                )

            info_io("\n".join(msg), flush_block=not (int(os.getenv("DEBUG", 0))))
            # Join publisher
            if done_count == self._n_consumers:
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
