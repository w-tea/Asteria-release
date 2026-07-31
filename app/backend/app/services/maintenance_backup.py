from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
from typing import Callable
from uuid import uuid4

from app.services.atomic_json import write_json_atomic
from app.services.data_contracts import (
    DataContractManifest,
    PathKind,
    StorageClass,
    current_data_contract,
    validate_owned_paths,
)
from app.services.maintenance_journal import MaintenanceJournalRepository
from app.services.schema_versions import CURRENT_SCHEMA_VERSIONS


_BACKUP_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MINIMUM_BACKUP_HEADROOM = 16 * 1024 * 1024


class MaintenanceBackupError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MaintenanceBackupSummary:
    backup_id: str
    created_at: str
    file_count: int
    total_bytes: int
    status: str = "completed"


@dataclass(frozen=True, slots=True)
class MaintenanceRestorePreview:
    backup_id: str
    created_at: str
    file_count: int
    total_bytes: int
    valid: bool


class MaintenanceBackupService:
    def __init__(
        self,
        data_dir: Path,
        backup_root: Path,
        *,
        data_contract: DataContractManifest | None = None,
    ) -> None:
        self.data_dir = data_dir.absolute()
        self.backup_root = backup_root.absolute()
        self.data_contract = data_contract or current_data_contract()
        self.journal = MaintenanceJournalRepository(self.backup_root / "maintenance-journal.json")
        self._restore_failure_injector: Callable[[str, int], None] | None = None

    def create_backup(self) -> MaintenanceBackupSummary:
        backup_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex}"
        self.backup_root.mkdir(parents=True, exist_ok=True)
        sources = self._durable_sources()
        expected_bytes = sum(self._source_size(path) for path in sources)
        required_bytes = expected_bytes + max(_MINIMUM_BACKUP_HEADROOM, expected_bytes // 20)
        if self._free_bytes(self.backup_root) < required_bytes:
            raise MaintenanceBackupError("insufficient_backup_space")

        staging = self.backup_root / f".{backup_id}.tmp"
        destination = self.backup_root / backup_id
        self.journal.record(
            operation_id=backup_id,
            kind="backup",
            phase="started",
            status="running",
            detail_code="backup_started",
        )
        try:
            files_root = staging / "files"
            files_root.mkdir(parents=True)
            records: list[dict[str, object]] = []
            for source in sources:
                relative_path = source.relative_to(self.data_dir).as_posix()
                target = files_root / PurePosixPath(relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if relative_path.casefold() == "asteria.db":
                    self._copy_sqlite(source, target)
                else:
                    self._copy_regular_file(source, target)
                records.append(self._file_record(target, relative_path))
            records.sort(key=lambda item: str(item["relative_path"]).casefold())
            created_at = datetime.now(UTC).isoformat()
            write_json_atomic(
                staging / "backup-manifest.json",
                {
                    "version": 1,
                    "backup_id": backup_id,
                    "created_at": created_at,
                    "source_data_contract": self.data_contract.contract_id,
                    "schemas": {
                        "application": CURRENT_SCHEMA_VERSIONS.application,
                        "settings": CURRENT_SCHEMA_VERSIONS.settings,
                        "database": CURRENT_SCHEMA_VERSIONS.database,
                    },
                    "files": records,
                },
            )
            self._validate_backup_directory(staging)
            staging.replace(destination)
            summary = MaintenanceBackupSummary(
                backup_id=backup_id,
                created_at=created_at,
                file_count=len(records),
                total_bytes=sum(int(item["size"]) for item in records),
            )
            self.journal.record(
                operation_id=backup_id,
                kind="backup",
                phase="completed",
                status="completed",
                detail_code="backup_completed",
            )
            return summary
        except MaintenanceBackupError:
            self._remove_staging(staging)
            self._record_failure(backup_id, "backup_failed")
            raise
        except (OSError, sqlite3.Error) as exc:
            self._remove_staging(staging)
            self._record_failure(backup_id, "backup_copy_failed")
            raise MaintenanceBackupError("backup_copy_failed") from exc
        except Exception as exc:
            self._remove_staging(staging)
            self._record_failure(backup_id, "backup_failed")
            raise MaintenanceBackupError("backup_failed") from exc

    def list_backups(self) -> list[MaintenanceBackupSummary]:
        if not self.backup_root.exists():
            return []
        summaries: list[MaintenanceBackupSummary] = []
        for path in self.backup_root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                manifest = self._read_manifest(path)
                summaries.append(self._summary(manifest))
            except MaintenanceBackupError:
                continue
        return sorted(summaries, key=lambda item: item.created_at, reverse=True)

    def preview_restore(self, backup_id: str) -> MaintenanceRestorePreview:
        backup_dir = self._backup_dir(backup_id)
        manifest = self._validate_backup_directory(backup_dir)
        summary = self._summary(manifest)
        return MaintenanceRestorePreview(
            backup_id=summary.backup_id,
            created_at=summary.created_at,
            file_count=summary.file_count,
            total_bytes=summary.total_bytes,
            valid=True,
        )

    def restore_backup(self, backup_id: str) -> MaintenanceBackupSummary:
        backup_dir = self._backup_dir(backup_id)
        manifest = self._validate_backup_directory(backup_dir)
        summary = self._summary(manifest)
        operation_id = f"restore-{uuid4().hex}"
        rollback = self.backup_root / f".{operation_id}.tmp"
        records = list(manifest["files"])
        snapshots: dict[str, bool] = {}
        self.journal.record(
            operation_id=operation_id,
            kind="restore",
            phase="started",
            status="running",
            detail_code="restore_started",
        )
        try:
            for item in records:
                relative_path = self._safe_relative_path(item["relative_path"])
                destination = self.data_dir / relative_path
                snapshots[relative_path.as_posix()] = destination.exists()
                if destination.exists():
                    snapshot = rollback / relative_path
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    self._copy_regular_file(destination, snapshot)

            restored_count = 0
            for item in records:
                relative_path = self._safe_relative_path(item["relative_path"])
                source = backup_dir / "files" / relative_path
                destination = self.data_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._replace_file(source, destination)
                restored_count += 1
                if self._restore_failure_injector is not None:
                    self._restore_failure_injector(relative_path.as_posix(), restored_count)
            self._remove_staging(rollback)
            self.journal.record(
                operation_id=operation_id,
                kind="restore",
                phase="completed",
                status="completed",
                detail_code="restore_completed",
            )
            return summary
        except Exception as exc:
            try:
                self._rollback_restore(rollback, snapshots)
            except Exception as rollback_exc:
                self._record_failure(operation_id, "restore_rollback_failed", kind="restore")
                raise MaintenanceBackupError("restore_rollback_failed") from rollback_exc
            self._record_failure(operation_id, "restore_failed", kind="restore")
            raise MaintenanceBackupError("restore_failed") from exc

    def _durable_sources(self) -> list[Path]:
        try:
            owned = validate_owned_paths(self.data_contract, self.data_dir)
        except ValueError as exc:
            raise MaintenanceBackupError("backup_ownership_invalid") from exc
        sources: list[Path] = []
        for owned_path in owned:
            if owned_path.entry.storage_class is not StorageClass.DURABLE:
                continue
            path = owned_path.path
            if not path.exists():
                continue
            if owned_path.entry.path_kind is PathKind.FILE:
                if not path.is_file() or self._is_link(path):
                    raise MaintenanceBackupError("backup_ownership_invalid")
                sources.append(path)
                continue
            if not path.is_dir() or self._is_link(path):
                raise MaintenanceBackupError("backup_ownership_invalid")
            for root, directories, filenames in os.walk(path, followlinks=False):
                root_path = Path(root)
                for directory in list(directories):
                    if self._is_link(root_path / directory):
                        raise MaintenanceBackupError("backup_ownership_invalid")
                for filename in filenames:
                    file_path = root_path / filename
                    if self._is_link(file_path) or not file_path.is_file():
                        raise MaintenanceBackupError("backup_ownership_invalid")
                    sources.append(file_path)
        return sorted(sources, key=lambda path: path.relative_to(self.data_dir).as_posix().casefold())

    def _validate_backup_directory(self, backup_dir: Path) -> dict:
        manifest = self._read_manifest(backup_dir)
        files = manifest.get("files")
        if not isinstance(files, list):
            raise MaintenanceBackupError("backup_manifest_invalid")
        seen: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise MaintenanceBackupError("backup_manifest_invalid")
            relative = self._safe_relative_path(item.get("relative_path"))
            key = relative.as_posix().casefold()
            if key in seen:
                raise MaintenanceBackupError("backup_manifest_invalid")
            seen.add(key)
            source = backup_dir / "files" / relative
            if not source.is_file() or self._is_link(source):
                raise MaintenanceBackupError("backup_file_missing")
            if source.stat().st_size != item.get("size") or self._hash_file(source) != item.get("sha256"):
                raise MaintenanceBackupError("backup_hash_mismatch")
            if relative.as_posix().casefold() == "asteria.db":
                self._verify_sqlite(source)
        return manifest

    def _read_manifest(self, backup_dir: Path) -> dict:
        try:
            payload = json.loads((backup_dir / "backup-manifest.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise MaintenanceBackupError("maintenance_backup_not_found") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise MaintenanceBackupError("backup_manifest_invalid") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise MaintenanceBackupError("backup_manifest_invalid")
        return payload

    def _backup_dir(self, backup_id: str) -> Path:
        if not _BACKUP_ID.fullmatch(backup_id):
            raise MaintenanceBackupError("maintenance_backup_not_found")
        path = self.backup_root / backup_id
        if not path.is_dir() or self._is_link(path):
            raise MaintenanceBackupError("maintenance_backup_not_found")
        return path

    def _summary(self, manifest: dict) -> MaintenanceBackupSummary:
        files = manifest["files"]
        return MaintenanceBackupSummary(
            backup_id=str(manifest["backup_id"]),
            created_at=str(manifest["created_at"]),
            file_count=len(files),
            total_bytes=sum(int(item["size"]) for item in files),
        )

    def _rollback_restore(self, rollback: Path, snapshots: dict[str, bool]) -> None:
        for relative_text, existed in snapshots.items():
            relative = PurePosixPath(relative_text)
            destination = self.data_dir / relative
            snapshot = rollback / relative
            if existed:
                self._replace_file(snapshot, destination)
            else:
                destination.unlink(missing_ok=True)
        self._remove_staging(rollback)

    def _copy_sqlite(self, source: Path, destination: Path) -> None:
        with closing(sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)) as source_connection:
            with closing(sqlite3.connect(destination)) as destination_connection:
                source_connection.backup(destination_connection)
                destination_connection.commit()
        self._verify_sqlite(destination)

    def _verify_sqlite(self, path: Path) -> None:
        try:
            with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
                result = connection.execute("pragma integrity_check").fetchone()
        except sqlite3.Error as exc:
            raise MaintenanceBackupError("backup_database_corrupt") from exc
        if result is None or result[0] != "ok":
            raise MaintenanceBackupError("backup_database_corrupt")

    def _copy_regular_file(self, source: Path, destination: Path) -> None:
        shutil.copy2(source, destination)

    def _replace_file(self, source: Path, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.restore.tmp")
        try:
            self._copy_regular_file(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _file_record(self, path: Path, relative_path: str) -> dict[str, object]:
        return {
            "relative_path": relative_path,
            "size": path.stat().st_size,
            "sha256": self._hash_file(path),
        }

    def _hash_file(self, path: Path) -> str:
        digest = sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _source_size(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError as exc:
            raise MaintenanceBackupError("backup_source_unreadable") from exc

    def _free_bytes(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def _safe_relative_path(self, value: object) -> PurePosixPath:
        if not isinstance(value, str):
            raise MaintenanceBackupError("backup_manifest_invalid")
        relative = PurePosixPath(value.replace("\\", "/"))
        if (
            relative.is_absolute()
            or value == ""
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise MaintenanceBackupError("backup_manifest_invalid")
        return relative

    def _is_link(self, path: Path) -> bool:
        try:
            is_junction = getattr(os.path, "isjunction", None)
            return path.is_symlink() or bool(is_junction is not None and is_junction(path))
        except OSError as exc:
            raise MaintenanceBackupError("backup_ownership_invalid") from exc

    def _remove_staging(self, path: Path) -> None:
        if path.parent != self.backup_root or not path.name.startswith("."):
            raise MaintenanceBackupError("backup_cleanup_refused")
        shutil.rmtree(path, ignore_errors=True)

    def _record_failure(self, operation_id: str, detail_code: str, *, kind: str = "backup") -> None:
        try:
            self.journal.record(
                operation_id=operation_id,
                kind=kind,  # type: ignore[arg-type]
                phase="failed",
                status="failed",
                detail_code=detail_code,
            )
        except (OSError, ValueError):
            pass
