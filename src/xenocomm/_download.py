import urllib.request
from pathlib import Path

RELEASE = "v2"
FILES = (
    "mouse_lr.parquet",
    "mouse_rtg.parquet",
    "human_mouse_orthologs.parquet",
)


def ensure_database() -> Path:
    directory = Path.cwd() / ".xenocomm" / RELEASE
    if all((directory / name).is_file() for name in FILES):
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/bwalker1/xenocomm/releases/download/db-{RELEASE}"
    for name in FILES:
        print(f"Downloading {name}")
        urllib.request.urlretrieve(f"{base}/{name}", directory / name)
    return directory
