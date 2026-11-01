from __future__ import annotations

from collections.abc import Callable
from threading import Condition, Lock
from types import TracebackType
from typing import Self

from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send


class ReferenceMutationLease:
    def __init__(self, release: Callable[[], None]) -> None:
        self._release_callback = release
        self._release_lock = Lock()
        self._released = False

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._release_callback()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.release()


class ReferenceMutationCoordinator:
    def __init__(self) -> None:
        self._condition = Condition()
        self._publishers = 0
        self._cleanup_active = False

    def acquire_publisher(self) -> ReferenceMutationLease:
        with self._condition:
            while self._cleanup_active:
                self._condition.wait()
            self._publishers += 1
        return ReferenceMutationLease(self._release_publisher)

    def try_acquire_cleanup(self) -> ReferenceMutationLease | None:
        with self._condition:
            if self._cleanup_active or self._publishers:
                return None
            self._cleanup_active = True
        return ReferenceMutationLease(self._release_cleanup)

    def _release_publisher(self) -> None:
        with self._condition:
            self._publishers -= 1
            if self._publishers == 0:
                self._condition.notify_all()

    def _release_cleanup(self) -> None:
        with self._condition:
            self._cleanup_active = False
            self._condition.notify_all()


reference_mutation_coordinator = ReferenceMutationCoordinator()


class ReferenceMutationMiddleware:
    """Exclude asset cleanup without wrapping the application's ASGI channels."""

    def __init__(self, app: ASGIApp, *, coordinator: ReferenceMutationCoordinator) -> None:
        self.app = app
        self.coordinator = coordinator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope['type'] != 'http'
                or scope['method'] not in {'POST', 'PUT', 'PATCH', 'DELETE'}
                or scope['path'] == '/api/assets/lifecycle/cleanup'):
            await self.app(scope, receive, send)
            return
        lease = await run_in_threadpool(self.coordinator.acquire_publisher)
        try:
            await self.app(scope, receive, send)
        finally:
            lease.release()
