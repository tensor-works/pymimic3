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
    is_finished: bool = False
    _store_total: bool = True
    # These are discretizer only settings
    time_step_size: int = None
    start_at_zero: bool = None
    impute_strategy: str = None
    mode: str = None

    def __init__(self, num_subjects: int = None, subject_ids: list = None, **kwargs):
        self._lock = None
        # Continue processing if num subjects is not reached
        if num_subjects is not None and len(self.subjects) - 1 < num_subjects:
            self.is_finished = False
            self.num_subjects = num_subjects
        # Continue processing if num subjects switche to None
        elif self.num_subjects is not None and num_subjects is None:
            self.is_finished = False
            self.num_subjects = num_subjects

        if subject_ids is not None:
            unprocessed_subjects = set(subject_ids) - set(self.subjects.keys())
            if unprocessed_subjects:
                self.is_finished = False

        # The impute startegies of the discretizer might change
        # In this case we rediscretize the data
        if kwargs:
            for attribute in ["time_step_size", "start_at_zero", "impute_strategy", "mode"]:
                if attribute in kwargs:
                    if getattr(self, attribute) is not None and getattr(
                            self, attribute) != kwargs[attribute]:
                        self.reset()
                    setattr(self, attribute, kwargs[attribute])

    @property
    def subject_ids(self) -> list:
        if hasattr(self, "_progress"):
            return [
                subject_id for subject_id in self._read("subjects").keys() if subject_id != "total"
            ]
        return list()

    @property
    def stay_ids(self) -> list:
        if hasattr(self, "_progress"):
            return [
                stay_id for subject_id, subject_data in self._read("subjects").items()
                if subject_id != "total" for stay_id in subject_data.keys() if stay_id != "total"
            ]
        return list()

    @property
    def samples(self) -> int:
        if hasattr(self, "_progress"):
            return sum([
                subject_data["total"]
                for subject_id, subject_data in self._read("subjects").items()
                if subject_id != "total"
            ])
        return 0

    def reset(self):
        self.subjects = {}
        self.is_finished = False
        self.num_subjects = None


@storable
class DataSplitTracker():
    # Targets
    test_size: float = None
    val_size: float = None
    train_size: float = None
    subjects: dict = {}
    # Demographic settings
    demographic_filter: dict = None
    demographic_split: dict = None
    # Results
    split: dict = {}
    ratios: dict = {}
    is_finished: bool = False

    def __init__(self,
                 tracker: PreprocessingTracker,
                 test_size: float = 0.0,
                 val_size: float = 0.0,
                 demographic_filter: dict = None,
                 demographic_split: dict = None):
        """_summary_

        Args:
            tracker (PreprocessingTracker): _description_
            test_size (float, optional): _description_. Defaults to 0.0.
            val_size (float, optional): _description_. Defaults to 0.0.
            demographic_filter (dict, optional): _description_. Defaults to None.
            demographic_split (dict, optional): _description_. Defaults to None.
        """
        if self.is_finished:
            # Reset if changed
            if self.test_size != test_size:
                self.reset()
            elif self.val_size != val_size:
                self.reset()
            elif self.demographic_filter != demographic_filter:
                self.reset()
            elif self.demographic_split != demographic_split:
                self.reset()
        # Apply settings
        self.test_size = test_size
        self.val_size = val_size
        self.demographic_filter = demographic_filter
        self.demographic_split = demographic_split
        self.subjects = tracker.subjects

    def reset(self) -> None:
        # Reset the results
        self.is_finished = False
        self.ratios = {}
        self.split = {}

    @property
    def subject_ids(self) -> list:
        if hasattr(self, "_progress"):
            return [
                subject_id for subject_id in self._read("subjects").keys() if subject_id != "total"
            ]
        return list()

    @property
    def split_sets(self):
        if hasattr(self, "_progress"):
            return list(self.ratios.keys())
        return list()
