import shutil
from pathlib import Path
from tests.tsettings import *


def copy_dataset(folder: Path):
    if not Path(TEMP_DIR, folder).is_dir():
        source_path = Path(SEMITEMP_DIR, folder)
        target_path = Path(TEMP_DIR, folder)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(Path(SEMITEMP_DIR, folder)), str(Path(TEMP_DIR, folder)))
