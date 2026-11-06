from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, RLock
import time
from typing import Any, Iterator, Literal
from uuid import uuid4

from app.services.resource_lifecycle import (
    ResourceAdmissionLease,
    ResourceAdmissionService,
    resource_admission,
)

AdmissionKind = Literal["operation", "xyz"]


class ComfyAdmissionError(RuntimeError):
    pass


class ComfyAdmissionTimeoutError(ComfyAdmissionError):
    pass


class ComfyCleanupPendingError(ComfyAdmissionError):
    pass


@dataclass(frozen=True)
class ComfyAdmissionSnapshot:
    owner_kind: AdmissionKind | None
    cleanup_pending: bool


class ComfyAdmissionLease:
    def __init__(self, service: ComfyAdmissionService, token: str, kind: AdmissionKind) -> None:
        self._service = service
        self.token = token
        self.kind = kind
        self._released = False

    def __enter__(self) -> ComfyAdmissionLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release(defer_cleanup=self._service.cleanup_pending_for(self.token))

    def begin_cleanup(self) -> None:
        self._service.begin_cleanup(self.token)

    def release(self, *, defer_cleanup: bool = False) -> None:
        if self._released:
            return
        self._service.release(self.token, defer_cleanup=defer_cleanup)
        self._released = True


class ComfyAdmissionService:
    def __init__(self, resources: ResourceAdmissionService | None = None) -> None:
        self._condition = Condition(RLock())
        self._resources = resources or ResourceAdmissionService()
        self._owner_token: str | None = None
        self._owner_kind: AdmissionKind | None = None
        self._cleanup_pending_token: str | None = None
        self._resource_leases: dict[str, ResourceAdmissionLease] = {}

    def acquire(self, kind: AdmissionKind, *, timeout_seconds: float | None = None) -> ComfyAdmissionLease:
        if kind not in {"operation", "xyz"}:
            raise ValueError(f"Unsupported Comfy admission kind: {kind}")
        deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, timeout_seconds)
        with self._condition:
            while self._owner_token is not None or self._cleanup_pending_token is not None:
                if timeout_seconds == 0:
                    self._raise_unavailable()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._raise_unavailable()
                self._condition.wait(timeout=remaining)
            token = uuid4().hex
            resource_lease = self._resources.acquire(
                ("comfy_gpu_exclusive",),
                operation_id=f"comfy:{kind}:{token}",
                timeout_seconds=0,
            )
            self._owner_token = token
            self._owner_kind = kind
            self._resource_leases[token] = resource_lease
            return ComfyAdmissionLease(self, token, kind)

    @contextmanager
    def borrow(self, token: str, *, required_kind: AdmissionKind | None = None) -> Iterator[None]:
        with self._condition:
            if token != self._owner_token:
                raise ComfyAdmissionError("Comfy admission lease is not active")
            if required_kind is not None and self._owner_kind != required_kind:
                raise ComfyAdmissionError("Comfy admission lease has the wrong kind")
        yield

    def begin_cleanup(self, token: str) -> None:
        with self._condition:
            if token != self._owner_token or self._owner_kind != "xyz":
                raise ComfyAdmissionError("Only the active XYZ lease can begin cleanup")
            self._cleanup_pending_token = token

    def acknowledge_cleanup(self, token: str) -> None:
        with self._condition:
            if self._cleanup_pending_token != token:
                raise ComfyAdmissionError("Comfy cleanup token is not pending")
            self._cleanup_pending_token = None
            resource_lease = self._resource_leases.pop(token, None)
            if resource_lease is not None:
                resource_lease.release()
            self._condition.notify_all()

    def complete_deferred_cleanup(self, token: str) -> None:
        self.acknowledge_cleanup(token)

    def cleanup_pending_for(self, token: str) -> bool:
        with self._condition:
            return self._cleanup_pending_token == token

    def release(self, token: str, *, defer_cleanup: bool = False) -> None:
        with self._condition:
            if token != self._owner_token:
                raise ComfyAdmissionError("Comfy admission lease is not active")
            if self._cleanup_pending_token == token and not defer_cleanup:
                raise ComfyCleanupPendingError("Comfy cleanup acknowledgement is pending")
            self._owner_token = None
            self._owner_kind = None
            if not defer_cleanup:
                resource_lease = self._resource_leases.pop(token, None)
                if resource_lease is not None:
                    resource_lease.release()
            self._condition.notify_all()

    def snapshot(self) -> ComfyAdmissionSnapshot:
        with self._condition:
            return ComfyAdmissionSnapshot(
                owner_kind=self._owner_kind,
                cleanup_pending=self._cleanup_pending_token is not None,
            )

    def _raise_unavailable(self) -> None:
        if self._cleanup_pending_token is not None:
            raise ComfyCleanupPendingError("Comfy cleanup acknowledgement is pending")
        raise ComfyAdmissionTimeoutError("Comfy admission timed out")


def acknowledge_comfy_cleanup(
    *,
    service: ComfyAdmissionService,
    lease: ComfyAdmissionLease,
    client: Any,
    fence_workflow: dict,
    timeout_seconds: float,
) -> str:
    lease.begin_cleanup()
    _request_and_wait_for_cleanup(client, fence_workflow, timeout_seconds)
    service.acknowledge_cleanup(lease.token)
    return "xyz_cache_released"


def acknowledge_deferred_comfy_cleanup(
    *,
    service: ComfyAdmissionService,
    cleanup_token: str,
    client: Any,
    fence_workflow: dict,
    timeout_seconds: float,
) -> str:
    if not service.cleanup_pending_for(cleanup_token):
        raise ComfyAdmissionError("Comfy cleanup token is not pending")
    _request_and_wait_for_cleanup(client, fence_workflow, timeout_seconds)
    service.complete_deferred_cleanup(cleanup_token)
    return "xyz_cache_released"


def _request_and_wait_for_cleanup(client: Any, fence_workflow: dict, timeout_seconds: float) -> None:
    client.free_memory()
    for _ in range(2):
        prompt_id = client.queue_prompt(fence_workflow)
        client.wait_for_history(prompt_id, timeout_seconds=timeout_seconds)


comfy_admission = ComfyAdmissionService(resource_admission)
