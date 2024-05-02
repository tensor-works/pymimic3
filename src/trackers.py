from pathlib import Path
from pathos.helpers import mp
from utils import load_json, update_json, write_json, dict_subset
from utils.IO import *


class DataSplitTracker():
    """_summary_
    """

    def __init__(self, task_path: Path, model_path: Path, test_size: float,
                 val_size: float) -> None:
        """_summary_

        Args:
            task_path (Path): _description_
        """
        self._task_path = task_path
        self._task_path.mkdir(parents=True, exist_ok=True)
        self._progress_file = Path(task_path, "progress.json")
        self._split_file = Path(model_path, "split.json")
        self._subjects = load_json(self._progress_file)["subjects"]

        if not self._split_file.is_file():
            self._split_info = {
                "finished": False,
                "split_settings": {
                    "test_size": test_size,
                    "val_size": val_size
                },
                "ratios": {},
                "counts": {},
                "subjects": {},
            }
            self._finished = False
            update_json(self._split_file, self._split_info)
        else:
            self._split_info = load_json(self._split_file)
            self._finished = self._split_info["finished"]

    def reset(self, test_size: float, val_size: float) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        self._split_info = {
            "finished": False,
            "split_settings": {
                "test_size": test_size,
                "val_size": val_size
            },
            "ratios": {},
            "counts": {},
            "subjects": {},
        }
        self._finished = False
        write_json(self._split_file, self._split_info)

    @property
    def ratios(self) -> dict:
        return self._split_info["ratios"]

    @ratios.setter
    def ratios(self, value) -> dict:
        self._split_info["ratios"] = value
        update_json(self._split_file, self._split_info)

    @property
    def subjects(self) -> dict:
        return self._split_info["subjects"]

    @property
    def counts(self) -> dict:
        return self._split_info["counts"]

    @property
    def test(self) -> list:
        """_summary_

        Returns:
            bool: _description_
        """
        name = "test"
        if name in self._split_info["subjects"].keys():
            return self._split_info["subjects"][name]
        return None

    @test.setter
    def test(self, value: str) -> None:
        """_summary_

        Args:
            value (str): _description_
        """
        name = "test"
        self._split_info["subjects"].update({name: value})
        self._split_info["counts"].update({name: len(value)})
        update_json(self._split_file, self._split_info)

    @property
    def train(self) -> list:
        """_summary_

        Returns:
            bool: _description_
        """
        name = "train"
        if name in self._split_info["subjects"].keys():
            return self._split_info["subjects"][name]
        return None

    @train.setter
    def train(self, value: str) -> None:
        """_summary_

        Args:
            value (str): _description_
        """
        name = "train"
        self._split_info["subjects"].update({name: value})
        self._split_info["counts"].update({name: len(value)})
        update_json(self._split_file, self._split_info)

    @property
    def validation(self) -> list:
        """_summary_

        Returns:
            bool: _description_
        """
        name = "validation"
        if name in self._split_info["subjects"].keys():
            return self._split_info["subjects"][name]
        return None

    @validation.setter
    def validation(self, value: str) -> None:
        """_summary_

        Args:
            value (str): _description_
        """
        name = "validation"
        self._split_info["subjects"].update({name: value})
        self._split_info["counts"].update({name: len(value)})
        update_json(self._split_file, self._split_info)

    @property
    def subjects(self) -> set:
        """_summary_

        Returns:
            bool: _description_
        """
        return set([subject for subject in self._subjects.keys()])

    @property
    def subject_data(self) -> list:
        """_summary_

        Returns:
            bool: _description_
        """
        return self._subjects

    @property
    def directories(self) -> set:
        """_summary_

        Returns:
            bool: _description_
        """
        return set([
            folder.name
            for folder in self._task_path.iterdir()
            if folder.is_dir() and folder.name.isnumeric()
        ])

    @property
    def finished(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        self._finished = load_json(self._split_file)["finished"]
        return self._finished

    @finished.setter
    def finished(self, value: bool) -> None:
        """_summary_

        Args:
            value (bool): _description_
        """
        self._finished = value
        update_json(self._split_file, {"finished": value})
