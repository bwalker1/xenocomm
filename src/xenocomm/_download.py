"""Lazy-download and cache the xenocomm LR/RTG database."""

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

from platformdirs import user_data_dir

DB_VERSION = "v1"
DB_BASE_URL = f"https://github.com/bwalker1/xenocomm/releases/download/db-{DB_VERSION}"

DB_FILES = {
    "mouse_lr.parquet": "98fab21ef0c1a2255264620f47a3c2a3ca5a1d332a275a3a0950f1e5565d285b",
    "mouse_rtg.parquet": "6400712e102e21523dab44debdffaf8b8e7389a7a95e2b45df7b2b868251aa33",
}


def _data_dir() -> Path:
    return Path(user_data_dir("xenocomm"))


def _verify_checksum(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected[:12]}…, got {actual[:12]}…"
        )


def _has_database(directory: Path) -> bool:
    return all((directory / name).exists() for name in DB_FILES)


def _download_file(url: str, dest: Path, name: str) -> None:
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    with (
        Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress,
        urllib.request.urlopen(url) as response,
    ):
        total = int(response.headers.get("Content-Length", 0))
        task = progress.add_task(f"Downloading {name}", total=total or None)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            while chunk := response.read(1 << 16):
                f.write(chunk)
                progress.update(task, advance=len(chunk))


def get_database_dir() -> Path | None:
    local = Path.cwd() / ".xenocomm"
    if _has_database(local):
        return local
    global_dir = _data_dir() / DB_VERSION
    if _has_database(global_dir):
        return global_dir
    return None


def ensure_database() -> Path:
    found = get_database_dir()
    if found is not None:
        return found

    target = _data_dir() / DB_VERSION
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading xenocomm database ({DB_VERSION}) to {target}")

    for name, expected_sha in DB_FILES.items():
        url = f"{DB_BASE_URL}/{name}"
        tmp = Path(tempfile.mktemp(dir=target, suffix=f".{name}.tmp"))
        try:
            _download_file(url, tmp, name)
            _verify_checksum(tmp, expected_sha)
            shutil.move(tmp, target / name)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    return target


def download_database(target_dir: Path | None = None) -> Path:
    """Download the xenocomm database, even if it already exists locally."""
    target = Path(target_dir) if target_dir else _data_dir() / DB_VERSION
    target.mkdir(parents=True, exist_ok=True)

    for name, expected_sha in DB_FILES.items():
        url = f"{DB_BASE_URL}/{name}"
        tmp = Path(tempfile.mktemp(dir=target, suffix=f".{name}.tmp"))
        try:
            _download_file(url, tmp, name)
            _verify_checksum(tmp, expected_sha)
            shutil.move(tmp, target / name)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    return target
