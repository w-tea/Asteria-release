import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )
        with temporary_path.open("r+b") as temporary_file:
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
        _sync_directory(path.parent)
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
