from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat


@dataclass(frozen=True)
class LogRotationResult:
    rotated: bool
    warning_code: str | None = None


class OwnedLogRotator:
    def __init__(
        self,
        logs_root: Path,
        *,
        max_bytes: int = 32 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        self.logs_root = logs_root
        self.max_bytes = max_bytes
        self.backup_count = backup_count

    def rotate(self, log_path: Path, *, runtime_stopped: bool) -> LogRotationResult:
        if not runtime_stopped:
            return LogRotationResult(False, "active_log_not_rotated")
        try:
            root = self.logs_root.resolve(strict=True)
            path = log_path.resolve(strict=True)
            path.relative_to(root)
            details = os.lstat(path)
            if (
                not path.is_file()
                or stat.S_ISLNK(details.st_mode)
                or getattr(details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                return LogRotationResult(False, "log_not_owned")
            if details.st_size <= self.max_bytes:
                return LogRotationResult(False)
            for generation in range(self.backup_count, 0, -1):
                source = path.with_name(f"{path.name}.{generation}")
                if generation == self.backup_count and source.exists():
                    source.unlink()
                previous = path if generation == 1 else path.with_name(f"{path.name}.{generation - 1}")
                if previous.exists():
                    os.replace(previous, source)
            path.touch()
            return LogRotationResult(True)
        except ValueError:
            return LogRotationResult(False, "log_not_owned")
        except OSError:
            return LogRotationResult(False, "log_rotation_failed")
