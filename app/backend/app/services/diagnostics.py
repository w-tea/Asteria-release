from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable
from uuid import uuid4

import psutil

from app.core.config import AppConfig
from app.schemas.operations import ACTIVE_OPERATION_STATUSES
from app.services.log_rotation import OwnedLogRotator
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_retention import OperationRetentionService
from app.services.preview_registry import PreviewRegistry
from app.services.resource_observation import ResourceObserver
from app.services.runtime_process import RuntimeProcessManager
from app.services.runtime_recovery import (
    RuntimeOwnership,
    RuntimeProcessSample,
    RuntimeRecoveryError,
    RuntimeRecoveryService,
)


EXPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
EXPORT_NAME_PATTERN = re.compile(r"^diagnostics-([0-9a-f]{32})\.json$")
EXPORT_MAX_COUNT = 20
EXPORT_MAX_AGE = timedelta(days=30)


class DiagnosticsError(RuntimeError):
    pass


class PsutilRuntimeInspector:
    def sample(self, pid: int) -> RuntimeProcessSample | None:
        try:
            process = psutil.Process(pid)
            return RuntimeProcessSample(
                pid=pid,
                executable=process.exe(),
                start_time=process.create_time(),
                root=process.cwd(),
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.Error, OSError):
            return None

    def stop(self, pid: int) -> bool:
        try:
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return not process.is_running()
        except psutil.NoSuchProcess:
            return True
        except (psutil.AccessDenied, psutil.Error, OSError):
            return False


class DiagnosticsService:
    def __init__(
        self,
        config: AppConfig,
        coordinator: OperationCoordinator,
        *,
        observer: ResourceObserver | None = None,
        inspector: PsutilRuntimeInspector | None = None,
        runtime_manager: RuntimeProcessManager | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.coordinator = coordinator
        self.observer = observer or ResourceObserver(config, coordinator)
        self.inspector = inspector or PsutilRuntimeInspector()
        self.runtime_manager = runtime_manager or RuntimeProcessManager(config)
        self.clock = clock or (lambda: datetime.now(UTC))

    def status(self) -> dict[str, Any]:
        resources = self._sanitized_resources(self.observer.snapshot())
        runtime = resources["runtime"]
        operations = resources["operations"]
        admission = resources["admission"]
        issue_codes: list[str] = []
        readiness = "ready"

        if runtime["running"] is False:
            readiness = "action_required"
            issue_codes.append("runtime_unavailable")
        elif runtime["healthy"] is False:
            readiness = "degraded"
            issue_codes.append("runtime_unhealthy")
        if runtime["running"] is True and runtime["owned"] is not True:
            readiness = "action_required"
            issue_codes.append("runtime_ownership_unverified")
        if admission["cleanup_pending"]:
            readiness = "action_required"
            issue_codes.append("runtime_cleanup_pending")

        active_count = int(operations["active_count"])
        restart_reason = None
        if active_count:
            restart_reason = "runtime_recovery_active_operations"
        elif runtime["running"] is not True:
            restart_reason = "runtime_unavailable"
        elif runtime["owned"] is not True:
            restart_reason = "runtime_ownership_unverified"

        return {
            "schema_version": 1,
            "generated_at": self._now().isoformat(),
            "readiness": readiness,
            "issue_codes": issue_codes,
            "summary": {
                "runtime_running": runtime["running"],
                "runtime_healthy": runtime["healthy"],
                "runtime_owned": runtime["owned"],
                "active_operations": active_count,
                "queued_operations": operations["queue_length"],
                "waiting_claims": admission["waiting_count"],
                "cleanup_pending": admission["cleanup_pending"],
            },
            "recovery": {
                "cleanup_owned_metadata": {
                    "available": True,
                    "reason_code": None,
                },
                "restart_runtime_clean": {
                    "available": restart_reason is None,
                    "reason_code": restart_reason,
                },
            },
        }

    def export(self) -> dict[str, Any]:
        export_id = uuid4().hex
        now = self._now()
        resources = self._sanitized_resources(self.observer.snapshot())
        report = {
            "schema_version": 1,
            "generated_at": now.isoformat(),
            "readiness": self._status_from_resources(resources),
            "resources": resources,
        }
        root = self._export_root(create=True)
        destination = root / f"diagnostics-{export_id}.json"
        temporary = root / f".diagnostics-{export_id}.tmp"
        encoded = (
            json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, destination)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise DiagnosticsError("diagnostics_export_failed") from exc
        self._cleanup_exports(now)
        if not destination.exists():
            raise DiagnosticsError("diagnostics_export_failed")
        return {
            "export_id": export_id,
            "filename": f"Asteria-diagnostics-{export_id}.json",
            "created_at": now.isoformat(),
            "size_bytes": len(encoded),
        }

    def export_path(self, export_id: str) -> Path:
        if not EXPORT_ID_PATTERN.fullmatch(export_id):
            raise DiagnosticsError("diagnostics_export_id_invalid")
        root = self._export_root(create=False)
        candidate = root / f"diagnostics-{export_id}.json"
        try:
            details = os.lstat(candidate)
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            attributes = getattr(details, "st_file_attributes", 0)
            if (
                not stat.S_ISREG(details.st_mode)
                or stat.S_ISLNK(details.st_mode)
                or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise DiagnosticsError("diagnostics_export_not_found")
        except (OSError, RuntimeError, ValueError) as exc:
            raise DiagnosticsError("diagnostics_export_not_found") from exc
        return resolved

    def cleanup_owned_metadata(self, *, runtime_stopped: bool = False) -> dict[str, Any]:
        retention = OperationRetentionService(self.coordinator.repository).cleanup()
        previews = PreviewRegistry(self.config.data_dir / "asteria.db").cleanup()
        log_results = []
        if runtime_stopped:
            rotator = OwnedLogRotator(self.config.comfy_dir)
            for name in ("comfy.out.log", "comfy.err.log"):
                log_results.append(
                    rotator.rotate(
                        self.config.comfy_dir / name,
                        runtime_stopped=True,
                    )
                )
        return {
            "status": "completed",
            "removed_operation_records": retention.removed_operations,
            "removed_preview_files": previews.deleted_preview_files,
            "reclaimed_preview_bytes": previews.reclaimed_preview_bytes,
            "rotated_log_files": sum(item.rotated for item in log_results),
            "warning_codes": sorted(
                {
                    item.warning_code
                    for item in log_results
                    if item.warning_code is not None
                }
            ),
        }

    def restart_runtime_clean(self) -> dict[str, Any]:
        ownership = self._load_runtime_ownership()
        active_count = self._active_operation_count()
        warnings: list[str] = []

        def cleanup() -> None:
            self.cleanup_owned_metadata(runtime_stopped=True)

        def start() -> None:
            status = self.runtime_manager.start()
            if status.pid is not None and not self._record_started_runtime(status.pid):
                warnings.append("runtime_ownership_record_unavailable")

        try:
            RuntimeRecoveryService(self.inspector).restart_clean(
                ownership,
                active_operation_count=active_count,
                cleanup=cleanup,
                start=start,
            )
        except RuntimeRecoveryError as exc:
            raise DiagnosticsError(str(exc)) from exc
        return {
            "status": "completed",
            "warning_codes": warnings,
        }

    def _status_from_resources(self, resources: dict[str, Any]) -> dict[str, Any]:
        runtime = resources["runtime"]
        admission = resources["admission"]
        issues: list[str] = []
        readiness = "ready"
        if runtime["running"] is False:
            readiness = "action_required"
            issues.append("runtime_unavailable")
        elif runtime["healthy"] is False:
            readiness = "degraded"
            issues.append("runtime_unhealthy")
        if runtime["running"] is True and runtime["owned"] is not True:
            readiness = "action_required"
            issues.append("runtime_ownership_unverified")
        if admission["cleanup_pending"]:
            readiness = "action_required"
            issues.append("runtime_cleanup_pending")
        return {"state": readiness, "issue_codes": issues}

    def _sanitized_resources(self, source: dict[str, Any]) -> dict[str, Any]:
        runtime = self._mapping(source.get("runtime"))
        backend = self._mapping(source.get("backend"))
        operations = self._mapping(source.get("operations"))
        admission = self._mapping(source.get("admission"))
        storage = self._mapping(source.get("storage"))
        vram = self._mapping(runtime.get("vram")) if runtime.get("vram") is not None else None
        disks = []
        for item in storage.get("disks", []):
            row = self._mapping(item)
            scope = row.get("scope")
            if scope not in {"data", "models", "outputs"}:
                continue
            disks.append(
                {
                    "scope": scope,
                    "total_bytes": self._optional_nonnegative_int(row.get("total_bytes")),
                    "used_bytes": self._optional_nonnegative_int(row.get("used_bytes")),
                    "free_bytes": self._optional_nonnegative_int(row.get("free_bytes")),
                }
            )
        active_states = {
            str(key): self._nonnegative_int(value)
            for key, value in self._mapping(operations.get("active_states")).items()
            if str(key) in ACTIVE_OPERATION_STATUSES
        }
        return {
            "runtime": {
                "running": self._optional_bool(runtime.get("running")),
                "healthy": self._optional_bool(runtime.get("healthy")),
                "connected": self._optional_bool(runtime.get("connected")),
                "owned": self._optional_bool(runtime.get("owned")),
                "memory_mb": self._optional_nonnegative_float(runtime.get("memory_mb")),
                "cpu_percent": self._optional_nonnegative_float(runtime.get("cpu_percent")),
                "uptime_seconds": self._optional_nonnegative_float(runtime.get("uptime_seconds")),
                "log_bytes": self._optional_nonnegative_int(runtime.get("log_bytes")),
                "vram": (
                    {
                        "total_bytes": self._optional_nonnegative_int(vram.get("total_bytes")),
                        "used_bytes": self._optional_nonnegative_int(vram.get("used_bytes")),
                        "free_bytes": self._optional_nonnegative_int(vram.get("free_bytes")),
                    }
                    if vram is not None
                    else None
                ),
            },
            "backend": {
                "memory_mb": self._optional_nonnegative_float(backend.get("memory_mb")),
                "cpu_percent": self._optional_nonnegative_float(backend.get("cpu_percent")),
                "uptime_seconds": self._optional_nonnegative_float(backend.get("uptime_seconds")),
            },
            "operations": {
                "active_count": self._nonnegative_int(operations.get("active_count")),
                "active_states": active_states,
                "terminal_count": self._nonnegative_int(operations.get("terminal_count")),
                "oldest_terminal_age_seconds": self._optional_nonnegative_float(
                    operations.get("oldest_terminal_age_seconds")
                ),
                "queue_length": self._nonnegative_int(operations.get("queue_length")),
            },
            "admission": {
                "active_count": self._nonnegative_int(admission.get("active_count")),
                "waiting_count": self._nonnegative_int(admission.get("waiting_count")),
                "cleanup_pending": bool(admission.get("cleanup_pending")),
            },
            "storage": {
                "preview_files": self._nonnegative_int(storage.get("preview_files")),
                "preview_bytes": self._nonnegative_int(storage.get("preview_bytes")),
                "temp_files": self._nonnegative_int(storage.get("temp_files")),
                "temp_bytes": self._nonnegative_int(storage.get("temp_bytes")),
                "log_files": self._nonnegative_int(storage.get("log_files")),
                "log_bytes": self._nonnegative_int(storage.get("log_bytes")),
                "disks": disks[:3],
            },
        }

    def _load_runtime_ownership(self) -> RuntimeOwnership:
        state_path = self.config.data_dir / "temp" / "launcher" / "launcher-state.json"
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8-sig"))
            if not self._same_path(payload["app_root"], self.config.root_dir):
                raise DiagnosticsError("runtime_ownership_unverified")
            records = [
                item
                for item in payload["processes"]
                if isinstance(item, dict) and item.get("name") == "comfyui"
            ]
            if len(records) != 1:
                raise DiagnosticsError("runtime_ownership_unverified")
            record = records[0]
            started = datetime.fromisoformat(
                str(record["started_at_utc"]).replace("Z", "+00:00")
            ).timestamp()
            return RuntimeOwnership(
                pid=int(record["pid"]),
                executable=str(Path(record["executable_path"]).resolve(strict=False)),
                start_time=started,
                root=str(self.config.comfy_dir.resolve(strict=False)),
            )
        except DiagnosticsError:
            raise
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise DiagnosticsError("runtime_ownership_unverified") from exc

    def _record_started_runtime(self, pid: int) -> bool:
        sample = self.inspector.sample(pid)
        if sample is None or sample.pid != pid:
            return False
        state_path = self.config.data_dir / "temp" / "launcher" / "launcher-state.json"
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8-sig"))
            if not self._same_path(payload["app_root"], self.config.root_dir):
                return False
            records = [
                item
                for item in payload.get("processes", [])
                if isinstance(item, dict) and item.get("name") != "comfyui"
            ]
            records.append(
                {
                    "name": "comfyui",
                    "pid": sample.pid,
                    "executable_path": sample.executable,
                    "started_at_utc": datetime.fromtimestamp(
                        sample.start_time,
                        tz=UTC,
                    ).isoformat(),
                }
            )
            payload["processes"] = records
            temporary = state_path.with_name(f".{state_path.name}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, state_path)
            return True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def _active_operation_count(self) -> int:
        observed = self._mapping(self.observer.snapshot().get("operations"))
        observed_count = self._nonnegative_int(observed.get("active_count"))
        records = self.coordinator.list(limit=1000)
        actual_count = sum(item.status in ACTIVE_OPERATION_STATUSES for item in records)
        return max(observed_count, actual_count)

    def _cleanup_exports(self, now: datetime) -> None:
        root = self._export_root(create=True)
        files: list[tuple[datetime, Path]] = []
        try:
            for path in root.iterdir():
                if not EXPORT_NAME_PATTERN.fullmatch(path.name):
                    continue
                details = os.lstat(path)
                attributes = getattr(details, "st_file_attributes", 0)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or stat.S_ISLNK(details.st_mode)
                    or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                ):
                    continue
                modified = datetime.fromtimestamp(details.st_mtime, tz=UTC)
                files.append((modified, path))
        except OSError:
            return
        newest = sorted(files, key=lambda item: (item[0], item[1].name), reverse=True)
        for index, (modified, path) in enumerate(newest):
            if index < EXPORT_MAX_COUNT and now - modified < EXPORT_MAX_AGE:
                continue
            try:
                path.unlink()
            except OSError:
                continue

    def _export_root(self, *, create: bool) -> Path:
        root = self.config.data_dir / "diagnostics"
        try:
            if create:
                root.mkdir(parents=True, exist_ok=True)
            details = os.lstat(root)
            attributes = getattr(details, "st_file_attributes", 0)
            if (
                not stat.S_ISDIR(details.st_mode)
                or stat.S_ISLNK(details.st_mode)
                or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise DiagnosticsError("diagnostics_export_root_unavailable")
            return root.resolve(strict=True)
        except DiagnosticsError:
            raise
        except (OSError, RuntimeError) as exc:
            raise DiagnosticsError("diagnostics_export_root_unavailable") from exc

    def _now(self) -> datetime:
        value = self.clock()
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    @staticmethod
    def _same_path(left: str | Path, right: str | Path) -> bool:
        try:
            return os.path.normcase(str(Path(left).resolve(strict=False))) == os.path.normcase(
                str(Path(right).resolve(strict=False))
            )
        except (OSError, RuntimeError):
            return False

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _nonnegative_int(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return max(0, int(value))

    @classmethod
    def _optional_nonnegative_int(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return cls._nonnegative_int(value)

    @staticmethod
    def _optional_nonnegative_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(0.0, float(value))
