"""
Tracking Module
===============

This module provides classes for tracking various stages of the data processing pipeline, including 
extraction, preprocessing, and data splitting. These classes help in maintaining a persistent state 
and ensure that the data processing is robust to interruptions, enabling restoration of state in case 
of crashes.

The following table describes the input and tracking details for each tracker in this module:

+------------------------+-------------------+----------------------------------+
| Tracker                | Input             | Tracking                         |
+========================+===================+==================================+
| ExtractionTracker      | Raw dataset       | Tracks number of events,         |
|                        |                   | samples, and extraction status.  |
|                        |                   | Ensures crash resilience and     |
|                        |                   | state restoration of the         |
|                        |                   | extraction process.              |
+------------------------+-------------------+----------------------------------+
| PreprocessingTracker   | Extracted data    | Tracks subjects, preprocessing   |
|                        |                   | steps, and status. Ensures crash |
|                        |                   | resilience and state restoration |
|                        |                   | of the preprocessing process.    |
+------------------------+-------------------+----------------------------------+
| DataSplitTracker       | Preprocessed data | Tracks split ratios, demographic |
|                        |                   | filters, and split status.       |
|                        |                   | Ensures crash resilience and     |
|                        |                   | state restoration of the data    |
|                        |                   | splitting process.               |
+------------------------+-------------------+----------------------------------+
"""

from typing import List
from settings import *
from utils.IO import *
from storable import storable
from utils.jsons import write_json
from pathlib import Path


@storable
class ExtractionTracker():
    """
    Tracks the extraction process of the dataset.

    This class keeps track of various aspects of the data extraction process in a persistent fashion, enabeling
    robustness to crashes and avoiding reprocessing already existing results. The tracking includes the number
    of events extracted from different files, the total number of samples, and the progress of the
    extraction for its different steps.
    """

    #: dict: A dictionary tracking the number of events extracted from specific files.
    start_event_rows: dict = {"OUTPUTEVENTS.csv": 0, "LABEVENTS.csv": 0, "CHARTEVENTS.csv": 0}
    #: int: The total number of samples extracted as timeseries.
    count_total_samples: int = 0
    #: bool: Flag indicating if by-subject information has been extracted.
    has_bysubject_info: bool = False
    #: bool: Flag indicating if episodic data has been extracted.
    has_episodic_data: bool = False
    #: bool: Flag indicating if timeseries data has been extracted.
    has_timeseries: bool = False
    #: bool: Flag indicating if subject events have been extracted.
    has_subject_events: bool = False
    #: bool: Flag indicating if ICU history has been extracted.
    has_icu_history: bool = False
    #: bool: Flag indicating if diagnoses data has been extracted.
    has_diagnoses: bool = False
    #: bool: Flag indicating if the extraction process is finished.
    finished: bool = False
    #: list: A list of subject IDs for tracking progress.
    subject_ids: list = list()  # Not extraction target but tracking
    #: int: The target number of samples for extraction.
    num_samples: int = None  # Extraction target
    #: int: The target number of subjects for extraction.
    num_subjects: int = None  # Extraction target

    def __init__(self,
                 num_samples: int = None,
                 num_subjects: int = None,
                 subject_ids: list = None,
                 *args,
                 **kwargs) -> None:
        # TODO: proof storables against concurrent access. Currently this has to be handles
        # externally.
        self._lock = None
        # If num samples has increase more samples need to be raised, if decreased value error
        if self.num_samples is not None and num_samples is None:
            # If changed to None extraction is carried out for all samples
            self.reset(flags_only=True)
            self.num_samples = num_samples
        elif num_samples is not None and self.count_total_samples < num_samples:
            self.reset(flags_only=True)
            self.num_samples = num_samples

        if num_subjects is not None and len(self.subject_ids) < num_subjects:
            self.reset(flags_only=True, reset_start_rows=True)
            self.num_subjects = num_subjects
        # Continue processing if num subjects switche to None
        elif self.num_subjects is not None and num_subjects is None:
            self.reset(flags_only=True, reset_start_rows=True)
            self.num_subjects = num_subjects

        if subject_ids is not None:
            unprocessed_subjects = set(subject_ids) - set(self.subject_ids)
            if unprocessed_subjects:
                self.reset(flags_only=True, reset_start_rows=True)

    def reset(self, flags_only: bool = False, reset_start_rows: bool = False) -> None:
        """
        Resets the tracker state.

        Parameters
        ----------
        flags_only : bool, optional
            If True, only reset flags; otherwise, reset all counts and lists.
        """
        if not flags_only:
            self.start_event_rows = {
                "OUTPUTEVENTS.csv": 0,
                "LABEVENTS.csv": 0,
                "CHARTEVENTS.csv": 0
            }
            self.count_total_samples = 0
            self.num_samples = None
            self.num_subjects = None
            self.subject_ids = list()

        if reset_start_rows:
            self.start_event_rows = {
                "OUTPUTEVENTS.csv": 0,
                "LABEVENTS.csv": 0,
                "CHARTEVENTS.csv": 0
            }

        # The other dfs are light weight and computed for all subjects
        self.has_episodic_data = False
        self.has_timeseries = False
        self.has_bysubject_info = False
        self.has_subject_events = False
        self.finished = False

    @property
    def is_finished(self):
        return self.finished

    @is_finished.setter
    def is_finished(self, value):
        assert isinstance(value, bool)
        self.finished = value
        if value:
            write_json(Path(str(self._path) + ".json"), self._read())


@storable
class PreprocessingTracker():
    """
    Tracks the preprocessing of the dataset.

    This class keeps track of the preprocessing steps for the dataset, including the number of
    subjects processed and various preprocessing settings. Can be used for preprocessing,
    discretization, and feature engineering.
    """

    #: dict: A dictionary tracking the subjects and their preprocessing status.
    subjects: dict = {}
    #: int: The target number of subjects for preprocessing.
    num_subjects: int = None
    #: bool: Flag indicating if the preprocessing is finished.
    finished: bool = False
    #: bool: Flag indicating if the total count should be stored.
    _store_total: bool = True
    #: int: Discretization only. The time step size used in preprocessing.
    time_step_size: int = None
    #: bool: Discretization only. Flag indicating if the time series should start at zero.
    start_at_zero: bool = None
    #: str: The strategy used for imputing missing data.
    impute_strategy: str = None
    #: str: Discretization only. Legacy or experimental mode.
    mode: str = None
    #: list: already create supervision modes
    supervision_modes: list = list()
    #: bool: Flag indicating if deep supervision is used.
    deep_supervision: bool = None
    #: bool: Flag indicating the rerun does not depend on existing subject data
    force_rerun: bool = False

    def __init__(self, **kwargs):
        # Do nothing on init since subject_ids and num_subjects are not yet available
        self._lock = None
        self.deep_supervision = kwargs.get("deep_supervision", None)

        # The impute startegies of the discretizer might change
        # In this case we rediscretize the data
        if kwargs:
            for attribute in ["time_step_size", "start_at_zero", "impute_strategy", "mode"]:
                if attribute in kwargs:
                    if getattr(self, attribute) is not None and getattr(
                            self, attribute) != kwargs[attribute]:
                        self.reset()
                        self.force_rerun = True
                    setattr(self, attribute, kwargs[attribute])

        if self.deep_supervision is not None and self.deep_supervision:
            if "deep_supervision" not in self.supervision_modes:
                self.reset()
                self.force_rerun = True
        elif self.deep_supervision is not None and not self.deep_supervision:
            if "no_deep_supervision" not in self.supervision_modes:
                self.reset()
                self.force_rerun = True

    def set_subject_ids(self, subject_ids: List[int]):
        """Set the subjects to be processed to see if reprocessing is necessary.

        Args
        ----
            subject_ids (List[int]): to be processed IDs
        """
        if subject_ids is not None:
            unprocessed_subjects = set(subject_ids) - set(self.subjects.keys())
            if unprocessed_subjects:
                self.is_finished = False

    def set_num_subjects(self, num_subjects: int):
        """Set the number of target processed subjects to see if reprocessing is necessary.

        Args
        ----
            num_subjects (int): number of to be processed subjects.
        """
        # Continue processing if num subjects is not reached
        if num_subjects is not None and len(self.subjects) - 1 < num_subjects:
            self.is_finished = False
            self.num_subjects = num_subjects
        # Continue processing if num subjects switche to None
        elif self.num_subjects is not None and num_subjects is None:
            self.is_finished = False
            self.num_subjects = num_subjects

    @property
    def is_finished(self):
        return self.finished

    @is_finished.setter
    def is_finished(self, value):
        assert isinstance(value, bool)
        self.finished = value
        if value:
            if self.deep_supervision is not None and self.deep_supervision:
                if "deep_supervision" not in self.supervision_modes:
                    self.supervision_modes.append("deep_supervision")
            elif self.deep_supervision is not None and not self.deep_supervision:
                if "no_deep_supervision" not in self.supervision_modes:
                    self.supervision_modes.append("no_deep_supervision")
            self.force_rerun = False
            self._to_dict()

    def _to_dict(self):
        write_json(Path(str(self._path) + ".json"), self._read())

    @property
    def subject_ids(self) -> List[int]:
        """
        Get the list of subject IDs.

        Returns
        -------
        list
            A list of subject IDs.
        """
        if hasattr(self, "_progress"):
            return [
                subject_id for subject_id in self._get_callback("subjects").keys()
                if subject_id != "total"
            ]
        return list()

    @property
    def stay_ids(self) -> List[int]:
        """
        Get the list of stay IDs.

        Returns
        -------
        list
            A list of stay IDs.
        """
        if hasattr(self, "_progress"):
            return [
                stay_id for subject_id, subject_data in self._get_callback("subjects").items()
                if subject_id != "total" for stay_id in subject_data.keys() if stay_id != "total"
            ]
        return list()

    @property
    def samples(self) -> int:
        """
        Get the total number of samples processed.

        Returns
        -------
        int
            The total number of samples processed.
        """
        if hasattr(self, "_progress"):
            return sum([
                subject_data["total"]
                for subject_id, subject_data in self._get_callback("subjects").items()
                if subject_id != "total"
            ])
        return 0

    def reset(self):
        """
        Resets the tracker state.
        """
        self.subjects = {}
        self.is_finished = False
        self.num_subjects = None


@storable
class DataSplitTracker():
    """
    Tracks the data splitting process.

    This class keeps track of the data splitting process, including the sizes of the train, 
    validation, and test sets, as well as demographic settings.
    """

    #: float: The proportion of the data to be used as the test set.
    test_size: float = None
    #: float: The proportion of the data to be used as the validation set.
    val_size: float = None
    #: float: The proportion of the data to be used as the training set.
    train_size: float = None
    #: dict: A dictionary tracking the subjects and their split status.
    subjects: dict = {}
    #: dict: A dictionary specifying demographic filters to be applied.
    demographic_filter: dict = None
    #: dict: A dictionary specifying demographic splits to be applied.
    demographic_split: dict = None
    #: dict: A dictionary tracking the split status of the data.
    split: dict = {}
    #: dict: A dictionary tracking the ratios of the splits.
    ratios: dict = {}
    #: bool: Flag indicating if the data splitting is finished.
    is_finished: bool = False

    def __init__(self,
                 tracker: PreprocessingTracker,
                 test_size: float = 0.0,
                 val_size: float = 0.0,
                 demographic_filter: dict = None,
                 demographic_split: dict = None):

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
        """
        Resets the tracker state.
        """
        # Reset the results
        self.is_finished = False
        self.ratios = {}
        self.split = {}

    @property
    def subject_ids(self) -> list:
        """
        Get the list of subject IDs.

        Returns
        -------
        list
            A list of subject IDs.
        """
        if hasattr(self, "_progress"):
            return [
                subject_id for subject_id in self._get_callback("subjects").keys()
                if subject_id != "total"
            ]
        return list()

    @property
    def split_sets(self):
        """
        Get the split sets.

        Returns
        -------
        list
            A list of split sets.
        """
        if hasattr(self, "_progress"):
            return list(self.ratios.keys())
        return list()
