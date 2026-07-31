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
from app.services.comfy_admission import comfy_admission
from app.services.operation_coordinator import OperationCoordinator
from app.services.resource_lifecycle import resource_admission
from app.services.runtime import ComfyClient
from app.services.runtime_process import RuntimeProcessManager


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
        state_path = self.config.data_dir / "temp" / "launcher" / "launcher-state.json"
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8-sig"))
            if os.path.normcase(str(Path(payload["app_root"]).resolve(strict=False))) != os.path.normcase(
                str(self.config.root_dir.resolve(strict=False))
            ):
                return False
            record = next(
                item
                for item in payload["processes"]
                if item.get("name") == "comfyui" and int(item.get("pid")) == pid
            )
            process = psutil.Process(pid)
            expected_time = datetime.fromisoformat(
                str(record["started_at_utc"]).replace("Z", "+00:00")
            ).timestamp()
            return (
                os.path.normcase(process.exe())
                == os.path.normcase(str(Path(record["executable_path"]).resolve(strict=False)))
                and abs(process.create_time() - expected_time) < 0.001
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            StopIteration,
            TypeError,
            json.JSONDecodeError,
            psutil.Error,
        ):
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
        return self._tree_summary(self.config.output_dir / "_previews")

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
            ("outputs", self.config.output_dir),
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
