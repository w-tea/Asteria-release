from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi import FastAPI

from app.services.atomic_json import write_json_atomic
from app.services.extension_loader import ExtensionLoadRecord, ExtensionLoader


EXTENSION_STARTUP_SCHEMA = "asteria.extension.startup.v1"


@dataclass(frozen=True, slots=True)
class ExtensionStartupResult:
    safe_mode: bool
    previous_incomplete: bool
    records: tuple[ExtensionLoadRecord, ...]
    journal_error: bool = False


class ExtensionStartupService:
    def __init__(self, loader: ExtensionLoader, *, journal_path: Path) -> None:
        self.loader = loader
        self.journal_path = journal_path

    def initialize(self, app: FastAPI, *, safe_mode: bool) -> ExtensionStartupResult:
        previous_incomplete = self._previous_incomplete()
        extension_root_exists = self.loader.repository.extension_root.exists()
        should_journal = extension_root_exists
        started_at = _timestamp()
        journal_error = False
        if should_journal:
            journal_error = not self._try_write_journal(
                status="loading",
                safe_mode=safe_mode,
                started_at=started_at,
                completed_at=None,
                records=(),
            )
        if safe_mode:
            records: tuple[ExtensionLoadRecord, ...] = ()
        elif not extension_root_exists:
            records = ()
        else:
            try:
                records = self.loader.load_into(app)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                records = (
                    ExtensionLoadRecord(
                        extension_id="asteria.extensions",
                        version=None,
                        status="failed",
                        error_code="extension_registry_failed",
                    ),
                )
        if should_journal:
            journal_error = not self._try_write_journal(
                status="ready",
                safe_mode=safe_mode,
                started_at=started_at,
                completed_at=_timestamp(),
                records=records,
            ) or journal_error
        return ExtensionStartupResult(
            safe_mode=safe_mode,
            previous_incomplete=previous_incomplete,
            records=records,
            journal_error=journal_error,
        )

    def _previous_incomplete(self) -> bool:
        if not self.journal_path.exists():
            return False
        try:
            payload = json.loads(self.journal_path.read_text(encoding="utf-8"))
            return not (
                isinstance(payload, dict)
                and payload.get("schema_version") == EXTENSION_STARTUP_SCHEMA
                and payload.get("status") == "ready"
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            return True

    def _try_write_journal(
        self,
        *,
        status: str,
        safe_mode: bool,
        started_at: str,
        completed_at: str | None,
        records: tuple[ExtensionLoadRecord, ...],
    ) -> bool:
        try:
            write_json_atomic(
                self.journal_path,
                {
                    "schema_version": EXTENSION_STARTUP_SCHEMA,
                    "status": status,
                    "safe_mode": safe_mode,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "records": [
                        {
                            "extension_id": record.extension_id,
                            "version": record.version,
                            "status": record.status,
                            "error_code": record.error_code,
                        }
                        for record in records
                    ],
                },
            )
            return True
        except OSError:
            return False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
