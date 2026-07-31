from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Any, Callable

from app.services.disk_reservation import DiskReservationError, DiskReservationService
from app.services.managed_artifacts import (
    ManagedArtifactCancelled,
    ManagedArtifactError,
    ManagedArtifactService,
    ManagedArtifactSpec,
)
from app.services.operation_executor import (
    OperationCancelled,
    OperationExecutionContext,
    OperationExecutionError,
    OperationExecutionResult,
)


DOWNLOAD_TELEMETRY_SQL = """
create table if not exists managed_download_telemetry (
    operation_id text primary key,
    bytes_downloaded integer not null,
    total_bytes integer not null,
    speed_bytes_per_second real,
    eta_seconds real,
    state text not null,
    updated_at real not null
)
"""


def create_managed_download_tables(connection: sqlite3.Connection) -> None:
    connection.execute(DOWNLOAD_TELEMETRY_SQL)


@dataclass(frozen=True)
class ManagedDownloadTelemetry:
    operation_id: str
    bytes_downloaded: int
    total_bytes: int
    percent: float
    speed_bytes_per_second: float | None
    eta_seconds: float | None
    state: str


class ManagedDownloadTelemetryRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        with sqlite3.connect(db_path) as connection:
            create_managed_download_tables(connection)

    def update(
        self,
        operation_id: str,
        *,
        downloaded: int,
        total: int,
        speed: float | None,
        eta: float | None,
        state: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                insert into managed_download_telemetry(
                    operation_id, bytes_downloaded, total_bytes,
                    speed_bytes_per_second, eta_seconds, state, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict(operation_id) do update set
                    bytes_downloaded = excluded.bytes_downloaded,
                    total_bytes = excluded.total_bytes,
                    speed_bytes_per_second = excluded.speed_bytes_per_second,
                    eta_seconds = excluded.eta_seconds,
                    state = excluded.state,
                    updated_at = excluded.updated_at
                """,
                (operation_id, downloaded, total, speed, eta, state, time.time()),
            )

    def get(self, operation_id: str) -> ManagedDownloadTelemetry | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "select * from managed_download_telemetry where operation_id = ?",
                (operation_id,),
            ).fetchone()
        if row is None:
            return None
        downloaded, total = int(row[1]), int(row[2])
        return ManagedDownloadTelemetry(
            operation_id=operation_id,
            bytes_downloaded=downloaded,
            total_bytes=total,
            percent=downloaded / total * 100 if total else 0,
            speed_bytes_per_second=float(row[3]) if row[3] is not None else None,
            eta_seconds=float(row[4]) if row[4] is not None else None,
            state=str(row[5]),
        )


_discard_lock = Lock()
_discard_requests: set[str] = set()


def request_partial_discard(operation_id: str) -> None:
    with _discard_lock:
        _discard_requests.add(operation_id)


def consume_partial_discard(operation_id: str) -> bool:
    with _discard_lock:
        existed = operation_id in _discard_requests
        _discard_requests.discard(operation_id)
        return existed


def partial_discard_requested(operation_id: str) -> bool:
    with _discard_lock:
        return operation_id in _discard_requests


class ManagedDownloadOperationExecutor:
    version = "managed-download-v1.1"

    def __init__(
        self,
        *,
        temp_root: Path,
        db_path: Path,
        disk_reservations: DiskReservationService | None = None,
        service_factory: Callable[..., ManagedArtifactService] = ManagedArtifactService,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.temp_root = temp_root
        self.telemetry = ManagedDownloadTelemetryRepository(db_path)
        self.disk_reservations = disk_reservations or DiskReservationService()
        self.service_factory = service_factory
        self.monotonic = monotonic

    def resource_claims(self, _snapshot: dict[str, Any]) -> tuple[str, ...]:
        return ("network_download", "disk_write_heavy")

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        spec = self._spec(context.request_snapshot)
        started = self.monotonic()
        last_time = started
        last_bytes = 0

        def progress(_component_id: str, downloaded: int, total: int) -> None:
            nonlocal last_time, last_bytes
            context.raise_if_cancelled()
            now = self.monotonic()
            elapsed = now - last_time
            speed = (downloaded - last_bytes) / elapsed if elapsed > 0 else None
            eta = (total - downloaded) / speed if speed and speed > 0 else None
            self.telemetry.update(
                context.operation.operation_id,
                downloaded=downloaded,
                total=total,
                speed=speed,
                eta=eta,
                state="running",
            )
            context.report_progress(downloaded, total, "bytes", phase="downloading")
            last_time, last_bytes = now, downloaded

        service = self.service_factory(
            self.temp_root,
            progress_callback=progress,
            cancel_requested=lambda: context.cancel_requested,
        )
        partial_ref = (f"download-partial:{spec.component_id}",)
        try:
            with self.disk_reservations.reserve(
                spec.destination.parent,
                spec.expected_size,
                owner_id=context.operation.operation_id,
            ):
                result = service.acquire(spec)
            self.telemetry.update(
                context.operation.operation_id,
                downloaded=spec.expected_size,
                total=spec.expected_size,
                speed=None,
                eta=0,
                state="completed",
            )
            return OperationExecutionResult(
                artifact_refs=(f"managed-component:{spec.component_id}",),
            )
        except (ManagedArtifactCancelled, OperationCancelled) as exc:
            discard = partial_discard_requested(context.operation.operation_id)
            if discard:
                service.discard_partial(spec)
            has_partial = service.has_valid_partial(spec)
            self.telemetry.update(
                context.operation.operation_id,
                downloaded=self._partial_size(service, spec),
                total=spec.expected_size,
                speed=None,
                eta=None,
                state="cancelled",
            )
            consume_partial_discard(context.operation.operation_id)
            raise OperationCancelled(
                artifact_refs=partial_ref if has_partial else (),
            ) from exc
        except DiskReservationError as exc:
            self.telemetry.update(
                context.operation.operation_id,
                downloaded=0,
                total=spec.expected_size,
                speed=None,
                eta=None,
                state="failed",
            )
            raise OperationExecutionError("disk_space_insufficient", retryable=True) from exc
        except ManagedArtifactError as exc:
            has_partial = service.has_valid_partial(spec)
            self.telemetry.update(
                context.operation.operation_id,
                downloaded=self._partial_size(service, spec),
                total=spec.expected_size,
                speed=None,
                eta=None,
                state="failed",
            )
            code = self._error_code(exc)
            raise OperationExecutionError(
                code,
                retryable=True,
                artifact_refs=partial_ref if has_partial else (),
            ) from exc

    @staticmethod
    def _spec(snapshot: dict[str, Any]) -> ManagedArtifactSpec:
        return ManagedArtifactSpec(
            manifest_version=int(snapshot["manifest_version"]),
            component_id=str(snapshot["component_id"]),
            source_url=str(snapshot["source_url"]),
            destination=Path(str(snapshot["destination"])),
            expected_size=int(snapshot["expected_size"]),
            checksum_sha256=str(snapshot["checksum_sha256"]),
            kind=str(snapshot["kind"]),
            strip_single_root=bool(snapshot.get("strip_single_root", False)),
        )

    @staticmethod
    def _partial_size(service: ManagedArtifactService, spec: ManagedArtifactSpec) -> int:
        try:
            return service.part_path_for(spec).stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _error_code(error: ManagedArtifactError) -> str:
        message = str(error)
        if "checksum" in message:
            return "managed_download_checksum_mismatch"
        if "size" in message:
            return "managed_download_size_mismatch"
        if "interrupted" in message:
            return "managed_download_interrupted"
        return "managed_download_failed"
