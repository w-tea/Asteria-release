from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class RuntimeOwnership:
    pid: int
    executable: str
    start_time: float
    root: str


@dataclass(frozen=True)
class RuntimeProcessSample:
    pid: int
    executable: str
    start_time: float
    root: str


class RuntimeProcessInspector(Protocol):
    def sample(self, pid: int) -> RuntimeProcessSample | None: ...

    def stop(self, pid: int) -> bool: ...


class RuntimeRecoveryError(RuntimeError):
    pass


class RuntimeRecoveryService:
    def __init__(self, inspector: RuntimeProcessInspector) -> None:
        self.inspector = inspector

    def ownership_matches(self, ownership: RuntimeOwnership) -> bool:
        try:
            sample = self.inspector.sample(ownership.pid)
        except (OSError, PermissionError):
            return False
        if sample is None:
            return False
        return (
            sample.pid == ownership.pid
            and os.path.normcase(sample.executable) == os.path.normcase(ownership.executable)
            and abs(sample.start_time - ownership.start_time) < 0.001
            and self._same_root(sample.root, ownership.root)
        )

    def restart_clean(
        self,
        ownership: RuntimeOwnership,
        *,
        active_operation_count: int,
        cleanup: Callable[[], None],
        start: Callable[[], None],
    ) -> None:
        if active_operation_count:
            raise RuntimeRecoveryError("runtime_recovery_active_operations")
        if not self.ownership_matches(ownership):
            raise RuntimeRecoveryError("runtime_ownership_unverified")
        try:
            stopped = self.inspector.stop(ownership.pid)
        except (OSError, PermissionError) as exc:
            raise RuntimeRecoveryError("runtime_stop_failed") from exc
        if not stopped:
            raise RuntimeRecoveryError("runtime_stop_failed")
        cleanup()
        start()

    @staticmethod
    def _same_root(left: str, right: str) -> bool:
        try:
            return os.path.normcase(str(Path(left).resolve(strict=False))) == os.path.normcase(
                str(Path(right).resolve(strict=False))
            )
        except (OSError, RuntimeError):
            return False
