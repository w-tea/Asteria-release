from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from threading import RLock
from typing import Callable
from uuid import uuid4


class DiskReservationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiskReservationSnapshot:
    reservation_id: str
    owner_id: str
    volume_root: str
    reserved_bytes: int


class DiskReservation:
    def __init__(
        self,
        service: DiskReservationService,
        reservation_id: str,
        owner_id: str,
        root: Path,
        volume_root: str,
        estimated_bytes: int,
        reserved_bytes: int,
    ) -> None:
        self._service = service
        self.reservation_id = reservation_id
        self.owner_id = owner_id
        self.root = root
        self.volume_root = volume_root
        self.estimated_bytes = estimated_bytes
        self.reserved_bytes = reserved_bytes
        self._released = False

    def __enter__(self) -> DiskReservation:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def revalidate(self, required_bytes: int | None = None) -> None:
        self._service.revalidate(
            self.reservation_id,
            required_bytes=self.estimated_bytes if required_bytes is None else required_bytes,
        )

    def publish(
        self,
        staging: Path,
        target: Path,
        *,
        register: Callable[[Path], None],
    ) -> None:
        try:
            details = staging.stat()
            if not staging.is_file() or details.st_size <= 0:
                raise DiskReservationError("artifact_staging_invalid")
            self.revalidate(details.st_size)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)
        except DiskReservationError:
            raise
        except OSError as exc:
            raise DiskReservationError("artifact_publish_failed") from exc
        try:
            register(target)
        except Exception as exc:
            try:
                target.unlink()
            except OSError:
                pass
            raise DiskReservationError("artifact_registration_failed") from exc

    def release(self) -> None:
        if self._released:
            return
        self._service.release(self.reservation_id)
        self._released = True


class DiskReservationService:
    def __init__(
        self,
        *,
        free_space: Callable[[Path], int] | None = None,
        safety_margin_bytes: int = 512 * 1024 * 1024,
        minimum_free_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        if safety_margin_bytes < 0 or minimum_free_bytes < 0:
            raise ValueError("disk_reservation_policy_invalid")
        self._free_space = free_space or (lambda path: shutil.disk_usage(path).free)
        self.safety_margin_bytes = safety_margin_bytes
        self.minimum_free_bytes = minimum_free_bytes
        self._reservations: dict[str, DiskReservationSnapshot] = {}
        self._lock = RLock()

    def reserve(
        self,
        target_root: Path,
        estimated_bytes: int,
        *,
        owner_id: str,
    ) -> DiskReservation:
        if estimated_bytes < 0:
            raise ValueError("disk_reservation_size_invalid")
        root = self._existing_anchor(target_root)
        volume_root = self._volume_identity(root)
        requested = estimated_bytes + self.safety_margin_bytes
        with self._lock:
            committed = sum(
                item.reserved_bytes
                for item in self._reservations.values()
                if item.volume_root == volume_root
            )
            free = self._sample_free(root)
            if free - committed - requested < self.minimum_free_bytes:
                raise DiskReservationError("disk_space_insufficient")
            reservation_id = uuid4().hex
            snapshot = DiskReservationSnapshot(
                reservation_id=reservation_id,
                owner_id=owner_id,
                volume_root=volume_root,
                reserved_bytes=requested,
            )
            self._reservations[reservation_id] = snapshot
        return DiskReservation(
            self,
            reservation_id,
            owner_id,
            root,
            volume_root,
            estimated_bytes,
            requested,
        )

    def revalidate(self, reservation_id: str, *, required_bytes: int) -> None:
        if required_bytes < 0:
            raise ValueError("disk_reservation_size_invalid")
        with self._lock:
            current = self._reservations.get(reservation_id)
            if current is None:
                raise DiskReservationError("disk_reservation_not_active")
            reserved_elsewhere = sum(
                item.reserved_bytes
                for key, item in self._reservations.items()
                if key != reservation_id and item.volume_root == current.volume_root
            )
            free = self._sample_free(Path(current.volume_root))
            needed = required_bytes + self.safety_margin_bytes
            if free - reserved_elsewhere - needed < self.minimum_free_bytes:
                raise DiskReservationError("disk_space_insufficient")

    def release(self, reservation_id: str) -> None:
        with self._lock:
            if reservation_id not in self._reservations:
                raise DiskReservationError("disk_reservation_not_active")
            del self._reservations[reservation_id]

    def snapshot(self) -> tuple[DiskReservationSnapshot, ...]:
        with self._lock:
            return tuple(self._reservations.values())

    def _sample_free(self, root: Path) -> int:
        try:
            value = int(self._free_space(root))
        except OSError as exc:
            raise DiskReservationError("disk_space_unavailable") from exc
        if value < 0:
            raise DiskReservationError("disk_space_unavailable")
        return value

    @staticmethod
    def _existing_anchor(path: Path) -> Path:
        current = Path(os.path.abspath(path))
        while not current.exists():
            if current.parent == current:
                raise DiskReservationError("disk_target_unavailable")
            current = current.parent
        return current

    @staticmethod
    def _volume_identity(path: Path) -> str:
        anchor = path.anchor or str(path.resolve(strict=True))
        return os.path.normcase(anchor)


disk_reservations = DiskReservationService()
