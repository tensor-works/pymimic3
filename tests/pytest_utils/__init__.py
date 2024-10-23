import shutil
from pathlib import Path
from tests.tsettings import *
from storable.mongo_dict import MongoDict


def copy_dataset(folder: Path):
    if not Path(TEMP_DIR, folder).is_dir():
        source_path = Path(SEMITEMP_DIR, folder)
        target_path = Path(TEMP_DIR, folder)
        MongoDict(Path(source_path, "progress")).copy(MongoDict(Path(target_path, "progress")))
        source_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(Path(SEMITEMP_DIR, folder)), str(Path(TEMP_DIR, folder)))
        Path(target_path, "progress").touch()
