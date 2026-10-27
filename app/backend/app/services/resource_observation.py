from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import psutil

from app.core.config import AppConfig
from app.schemas.operations import ACTIVE_OPERATION_STATUSES, TERMINAL_OPERATION_STATUSES
from app.schemas.resource_advisory import ResourceByteCapacity, ResourceObservationSnapshot
from app.services.comfy_admission import comfy_admission
from app.services.operation_coordinator import OperationCoordinator
from app.services.resource_lifecycle import resource_admission
from app.services.runtime import ComfyClient
from app.services.runtime_process import RuntimeProcessManager
from app.services.runtime_ownership import RuntimeOwnershipError, load_runtime_ownership, process_matches
from app.services.user_settings import resolve_current_output_root


class ResourceObserver:
    def __init__(
        self,
        config: AppConfig,
        coordinator: OperationCoordinator,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.coordinator = coordinator
        self.clock = clock

    def snapshot(self) -> dict[str, Any]:
        runtime_status = self._runtime_status()
        runtime_metrics = self._process_metrics(runtime_status.get("pid"))
        backend_metrics = self._process_metrics(os.getpid())
        operations = self.coordinator.list(limit=1000)
        active = [item for item in operations if item.status in ACTIVE_OPERATION_STATUSES]
        terminal = [item for item in operations if item.status in TERMINAL_OPERATION_STATUSES]
        oldest_terminal_age = None
        if terminal:
            oldest = min(item.completed_at or item.updated_at for item in terminal)
            oldest_terminal_age = max(
                0.0,
                datetime.now(UTC).timestamp() - oldest.timestamp(),
            )
        admission = resource_admission.snapshot()
        comfy = comfy_admission.snapshot()
        preview_files, preview_bytes = self._owned_preview_summary()
        temp_files, temp_bytes = self._tree_summary(self.config.data_dir / "temp")
        log_files, log_bytes = self._tree_summary(self.config.data_dir / "logs")
        return {
            "runtime": {
                "running": runtime_status["running"],
                "healthy": runtime_status["healthy"],
                "connected": runtime_status["healthy"],
                "owned": self._runtime_owned(runtime_status.get("pid")),
                **runtime_metrics,
                "log_bytes": self._runtime_log_bytes(),
                "vram": self._verified_vram(),
            },
            "backend": backend_metrics,
            "operations": {
                "active_count": len(active),
                "active_states": dict(sorted(Counter(item.status for item in active).items())),
                "terminal_count": len(terminal),
                "oldest_terminal_age_seconds": oldest_terminal_age,
                "queue_length": sum(item.status == "queued" for item in active),
            },
            "admission": {
                "active_count": len(admission.active),
                "waiting_count": len(admission.waiting),
                "cleanup_pending": comfy.cleanup_pending,
            },
            "storage": {
                "preview_files": preview_files,
                "preview_bytes": preview_bytes,
                "temp_files": temp_files,
                "temp_bytes": temp_bytes,
                "log_files": log_files,
                "log_bytes": log_bytes,
                "disks": self._disk_summary(),
            },
        }

    def advisory_capacity(self) -> ResourceObservationSnapshot:
        raw_vram = self._verified_vram()
        raw_ram = self._system_memory()
        disks = {str(item.get("scope")): item for item in self._disk_summary()}
        data_disk = disks.get("data", {})
        output_disk = disks.get("outputs", {})
        vram = None
        if raw_vram is not None:
            vram = ResourceByteCapacity(
                total_bytes=raw_vram["total_bytes"],
                available_bytes=raw_vram["free_bytes"],
            )
        ram = ResourceByteCapacity.model_validate(raw_ram) if raw_ram is not None else None
        return ResourceObservationSnapshot(
            vram=vram,
            ram=ram,
            temp_free_bytes=_optional_nonnegative_int(data_disk.get("free_bytes")),
            output_free_bytes=_optional_nonnegative_int(output_disk.get("free_bytes")),
        )

    def _runtime_status(self) -> dict[str, Any]:
        try:
            status = RuntimeProcessManager(self.config).status()
            return {
                "running": bool(status.running),
                "healthy": bool(status.healthy),
                "pid": status.pid,
            }
        except Exception:
            return {"running": None, "healthy": None, "pid": None}

    def _process_metrics(self, pid: int | None) -> dict[str, float | None]:
        if pid is None:
            return {
                "memory_mb": None,
                "cpu_percent": None,
                "uptime_seconds": None,
            }
        try:
            process = psutil.Process(pid)
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent(interval=None)
            uptime = max(0.0, self.clock() - process.create_time())
            return {
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": round(float(cpu_percent), 2),
                "uptime_seconds": round(uptime, 2),
            }
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.Error, OSError):
            return {
                "memory_mb": None,
                "cpu_percent": None,
                "uptime_seconds": None,
            }

    def _runtime_owned(self, pid: int | None) -> bool | None:
        if pid is None:
            return None
        try:
            ownership = load_runtime_ownership(self.config)
            if ownership is None:
                return False
            process = psutil.Process(ownership.pid)
            if not process_matches(process, ownership):
                return False
            return pid == ownership.pid or any(
                child.pid == pid and child.is_running()
                for child in process.children(recursive=True)
            )
        except (RuntimeOwnershipError, OSError, RuntimeError, ValueError, psutil.Error):
            return None

    def _verified_vram(self) -> dict[str, int] | None:
        try:
            stats = ComfyClient(self.config.comfy_base_url).system_stats()
            if not isinstance(stats, dict):
                return None
            devices = stats.get("devices")
            if not isinstance(devices, list) or not devices:
                return None
            totals = [
                (item.get("vram_total"), item.get("vram_free"))
                for item in devices
                if isinstance(item, dict)
            ]
            if not totals or any(
                isinstance(total, bool)
                or isinstance(free, bool)
                or not isinstance(total, int)
                or not isinstance(free, int)
                or total < 0
                or free < 0
                or free > total
                for total, free in totals
            ):
                return None
            total_bytes = sum(item[0] for item in totals)
            free_bytes = sum(item[1] for item in totals)
            return {
                "total_bytes": total_bytes,
                "free_bytes": free_bytes,
                "used_bytes": total_bytes - free_bytes,
            }
        except Exception:
            return None

    @staticmethod
    def _system_memory() -> dict[str, int] | None:
        try:
            memory = psutil.virtual_memory()
            total = memory.total
            available = memory.available
            if (
                isinstance(total, bool)
                or isinstance(available, bool)
                or not isinstance(total, int)
                or not isinstance(available, int)
                or total < 0
                or available < 0
                or available > total
            ):
                return None
            return {"total_bytes": total, "available_bytes": available}
        except (psutil.Error, OSError, RuntimeError, ValueError):
            return None

    def _runtime_log_bytes(self) -> int | None:
        total = 0
        found = False
        for path in (
            self.config.comfy_dir / "comfy.out.log",
            self.config.comfy_dir / "comfy.err.log",
        ):
            try:
                if path.is_file():
                    total += path.stat().st_size
                    found = True
            except OSError:
                return None
        return total if found else 0

    def _owned_preview_summary(self) -> tuple[int, int]:
        return self._tree_summary(
            resolve_current_output_root(self.config.data_dir) / "_previews"
        )

    @staticmethod
    def _tree_summary(root: Path, *, limit: int = 10_000) -> tuple[int, int]:
        try:
            resolved_root = root.resolve(strict=True)
            if not resolved_root.is_dir() or root.is_symlink():
                return 0, 0
        except (OSError, RuntimeError):
            return 0, 0
        files = total = 0
        try:
            for index, path in enumerate(resolved_root.rglob("*")):
                if index >= limit:
                    break
                try:
                    if path.is_file() and not path.is_symlink():
                        files += 1
                        total += path.stat().st_size
                except OSError:
                    continue
        except OSError:
            return files, total
        return files, total

    def _disk_summary(self) -> list[dict[str, int | str | None]]:
        summaries: list[dict[str, int | str | None]] = []
        for scope, path in (
            ("data", self.config.data_dir),
            ("models", self.config.model_dir),
            ("outputs", resolve_current_output_root(self.config.data_dir)),
        ):
            try:
                anchor = path
                while not anchor.exists() and anchor.parent != anchor:
                    anchor = anchor.parent
                usage = shutil.disk_usage(anchor)
                summaries.append(
                    {
                        "scope": scope,
                        "total_bytes": usage.total,
                        "used_bytes": usage.used,
                        "free_bytes": usage.free,
                    }
                )
            except OSError:
                summaries.append(
                    {
                        "scope": scope,
                        "total_bytes": None,
                        "used_bytes": None,
                        "free_bytes": None,
                    }
                )
        return summaries


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
