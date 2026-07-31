from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, RLock
import time
from typing import Iterable, Literal
from uuid import uuid4


ResourceClaim = Literal[
    "comfy_gpu_exclusive",
    "training_gpu_exclusive",
    "external_gpu_or_cpu",
    "cpu_heavy",
    "disk_write_heavy",
    "network_download",
    "maintenance_exclusive",
]

RESOURCE_LOCK_ORDER: tuple[ResourceClaim, ...] = (
    "maintenance_exclusive",
    "comfy_gpu_exclusive",
    "training_gpu_exclusive",
    "external_gpu_or_cpu",
    "cpu_heavy",
    "disk_write_heavy",
    "network_download",
)
_ORDER = {claim: index for index, claim in enumerate(RESOURCE_LOCK_ORDER)}
_GPU_CLAIMS = frozenset(
    {"comfy_gpu_exclusive", "training_gpu_exclusive", "external_gpu_or_cpu"}
)
_EXCLUSIVE_CLAIMS = frozenset(
    {"cpu_heavy", "disk_write_heavy", "network_download"}
)


class ResourceAdmissionError(RuntimeError):
    pass


class ResourceAdmissionTimeoutError(ResourceAdmissionError):
    pass


@dataclass(frozen=True)
class ResourceAdmissionEntry:
    operation_id: str
    claims: tuple[ResourceClaim, ...]


@dataclass(frozen=True)
class ResourceAdmissionSnapshot:
    active: tuple[ResourceAdmissionEntry, ...]
    waiting: tuple[ResourceAdmissionEntry, ...]


def normalize_claims(claims: Iterable[str]) -> tuple[ResourceClaim, ...]:
    unique = set(claims)
    invalid = unique.difference(_ORDER)
    if invalid:
        raise ValueError("resource_claim_invalid")
    return tuple(sorted(unique, key=_ORDER.__getitem__))  # type: ignore[return-value]


def claims_are_compatible(left: Iterable[str], right: Iterable[str]) -> bool:
    left_set = set(normalize_claims(left))
    right_set = set(normalize_claims(right))
    if "maintenance_exclusive" in left_set or "maintenance_exclusive" in right_set:
        return False
    if left_set & right_set & _EXCLUSIVE_CLAIMS:
        return False
    if left_set & _GPU_CLAIMS and right_set & _GPU_CLAIMS:
        return False
    return True


class ResourceAdmissionLease:
    def __init__(
        self,
        service: ResourceAdmissionService,
        token: str,
        operation_id: str,
        claims: tuple[ResourceClaim, ...],
    ) -> None:
        self._service = service
        self.token = token
        self.operation_id = operation_id
        self.claims = claims
        self._released = False

    def __enter__(self) -> ResourceAdmissionLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._service.release(self.token)
        self._released = True


@dataclass(frozen=True)
class _Waiter:
    token: str
    operation_id: str
    claims: tuple[ResourceClaim, ...]


class ResourceAdmissionService:
    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._active: dict[str, ResourceAdmissionEntry] = {}
        self._waiters: list[_Waiter] = []

    def acquire(
        self,
        claims: Iterable[str],
        *,
        operation_id: str,
        timeout_seconds: float | None = None,
    ) -> ResourceAdmissionLease:
        normalized = normalize_claims(claims)
        token = uuid4().hex
        waiter = _Waiter(token=token, operation_id=operation_id, claims=normalized)
        deadline = None if timeout_seconds is None else time.monotonic() + max(timeout_seconds, 0)
        with self._condition:
            self._waiters.append(waiter)
            try:
                while not self._can_admit(waiter):
                    if timeout_seconds == 0:
                        raise ResourceAdmissionTimeoutError("resource_admission_timed_out")
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise ResourceAdmissionTimeoutError("resource_admission_timed_out")
                    self._condition.wait(timeout=remaining)
                self._waiters.remove(waiter)
                self._active[token] = ResourceAdmissionEntry(operation_id, normalized)
                self._condition.notify_all()
                return ResourceAdmissionLease(self, token, operation_id, normalized)
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._condition.notify_all()
                raise

    def release(self, token: str) -> None:
        with self._condition:
            if token not in self._active:
                raise ResourceAdmissionError("resource_admission_lease_not_active")
            del self._active[token]
            self._condition.notify_all()

    def snapshot(self) -> ResourceAdmissionSnapshot:
        with self._condition:
            return ResourceAdmissionSnapshot(
                active=tuple(self._active.values()),
                waiting=tuple(
                    ResourceAdmissionEntry(item.operation_id, item.claims)
                    for item in self._waiters
                ),
            )

    def can_admit(
        self,
        claims: Iterable[str],
        *,
        earlier_waiting_claims: Iterable[Iterable[str]] = (),
    ) -> bool:
        normalized = normalize_claims(claims)
        with self._condition:
            active = tuple(item.claims for item in self._active.values())
        return all(claims_are_compatible(normalized, item) for item in (*active, *earlier_waiting_claims))

    def _can_admit(self, waiter: _Waiter) -> bool:
        if any(
            not claims_are_compatible(waiter.claims, item.claims)
            for item in self._active.values()
        ):
            return False
        for earlier in self._waiters:
            if earlier.token == waiter.token:
                break
            if not claims_are_compatible(waiter.claims, earlier.claims):
                return False
        return True


resource_admission = ResourceAdmissionService()
