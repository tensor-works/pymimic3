from typing import List, Tuple
from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """_summary_
    """

    @abstractmethod
    def __init__(self) -> None:
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        ...

    @property
    @abstractmethod
    def subjects(self) -> List[int]:
        ...

    @abstractmethod
    def transform(self, *args, **kwargs):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        ...

    @abstractmethod
    def transform_subject(self, subject_id: int) -> Tuple[dict, dict, dict]:
        ...

    @abstractmethod
    def save_data(self, subject_ids: list = None) -> None:
        ...
