from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Literal

from app.services.atomic_json import write_json_atomic


_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
MaintenanceKind = Literal["backup", "restore", "migration", "update"]
MaintenanceStatus = Literal["running", "completed", "failed", "recovery_required"]


@dataclass(frozen=True, slots=True)
class MaintenanceJournalRecord:
    version: int
    operation_id: str
    kind: MaintenanceKind
    phase: str
    status: MaintenanceStatus
    detail_code: str
    updated_at: str


class MaintenanceJournalRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> MaintenanceJournalRecord | None:
        try:
            if not self.path.exists():
                return None
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                return None
            return MaintenanceJournalRecord(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def record(
        self,
        *,
        operation_id: str,
        kind: MaintenanceKind,
        phase: str,
        status: MaintenanceStatus,
        detail_code: str,
    ) -> MaintenanceJournalRecord:
        for value, label in (
            (operation_id, "operation id"),
            (phase, "phase"),
            (detail_code, "detail code"),
        ):
            if not _SAFE_TOKEN.fullmatch(value):
                raise ValueError(f"maintenance {label} is invalid")
        record = MaintenanceJournalRecord(
            version=1,
            operation_id=operation_id,
            kind=kind,
            phase=phase,
            status=status,
            detail_code=detail_code,
            updated_at=datetime.now(UTC).isoformat(),
        )
        write_json_atomic(self.path, asdict(record))
        return record
