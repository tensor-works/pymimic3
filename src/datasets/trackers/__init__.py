from pathlib import Path
from utils.IO import *
from .storable import storable


@storable
class ExtractionTracker():
    """_summary_
    """
    count_subject_events: dict = {"OUTPUTEVENTS.csv": 0, "LABEVENTS.csv": 0, "CHARTEVENTS.csv": 0}
    count_total_samples: int = 0
    has_bysubject_info: bool = False
    has_episodic_data: bool = False
    has_timeseries: bool = False
    has_subject_events: bool = False
    has_icu_history: bool = False
    has_diagnoses: bool = False
    is_finished: bool = False
    subject_ids: list = list()  # Not extraction target but tracking
    num_samples: int = None  # Extraction target
    num_subjects: int = None  # Extraction target

    def __init__(self,
                 num_samples: int = None,
                 num_subjects: int = None,
                 subject_ids: list = None,
                 *args,
                 **kwargs) -> None:

        # If num samples has increase more samples need to be raised, if decreased value error
        if self.num_samples is not None and num_samples is None:
            # If changed to None extraction is carried out for all samples
            self.reset(flags_only=True)

        if num_samples is not None and self.count_total_samples < num_samples:
            self.reset(flags_only=True)
            self.num_samples = num_samples
        elif self.num_samples is not None and num_samples is None:
            self.reset(flags_only=True)
            self.num_samples = num_samples

        if num_subjects is not None and len(self.subject_ids) < num_subjects:
            self.reset(flags_only=True)
            self.num_subjects = num_subjects
        # Continue processing if num subjects switche to None
        elif self.num_subjects is not None and num_subjects is None:
            self.reset(flags_only=True)
            self.num_subjects = num_subjects

        if subject_ids is not None:
            unprocessed_subjects = set(subject_ids) - set(self.subject_ids)
            if unprocessed_subjects:
                self.reset(flags_only=True)

    def reset(self, flags_only: bool = False):
        if not flags_only:
            self.count_subject_events = {
                "OUTPUTEVENTS.csv": 0,
                "LABEVENTS.csv": 0,
                "CHARTEVENTS.csv": 0
            }
            self.count_total_samples = 0
            self.num_samples = None
            self.num_subjects = None
            self.subject_ids = list()
        # The other dfs are light weight and computed for all subjects
        self.has_episodic_data = False
        self.has_timeseries = False
        self.has_bysubject_info = False
        self.has_subject_events = False
        self.is_finished = False


@storable
class PreprocessingTracker():
    """_summary_
    """
    subjects: dict = {}
    num_subjects: int = None
    finished: bool = False
    _store_total: bool = True

    def __init__(self, num_subjects: int = None, subject_ids: list = None):
        self._lock = None
        # Continue processing if num subjects is not reached
        if num_subjects is not None and len(self.subjects) - 1 < num_subjects:
            self.finished = False
            self.num_subjects = num_subjects
        # Continue processing if num subjects switche to None
        elif self.num_subjects is not None and num_subjects is None:
            self.finished = False
            self.num_subjects = num_subjects

        if subject_ids is not None:
            unprocessed_subjects = set(subject_ids) - set(self.subjects.keys())
            if unprocessed_subjects:
                self.finished = False

    @property
    def subject_ids(self) -> list:
        if hasattr(self, "_progress"):
            return [
                subject_id for subject_id in self._progress.get("subjects", {}).keys()
                if subject_id != "total"
            ]
        return list()

    @property
    def stay_ids(self) -> list:
        if hasattr(self, "_progress"):
            return [
                stay_id for subject_id, subject_data in self._progress.get("subjects", {}).items()
                if subject_id != "total" for stay_id in subject_data.keys() if stay_id != "total"
            ]
        return list()

    @property
    def samples(self) -> int:
        if hasattr(self, "_progress"):
            return sum([
                subject_data["total"]
                for subject_id, subject_data in self._progress.get("subjects", {}).items()
                if subject_id != "total"
            ])
        return 0

    def reset(self):
        self.subjects = {}
        self.finished = False
        self.num_subjects = None
