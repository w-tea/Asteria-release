from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import re
from threading import Condition, Event, Lock, Thread
from time import sleep
from dataclasses import dataclass
from typing import Any, Mapping

from app.schemas.operations import OperationRecord, TERMINAL_OPERATION_STATUSES
from app.services.operation_executor import (
    OperationCancelled,
    OperationExecutionContext,
    OperationExecutionError,
    OperationExecutionResult,
    OperationExecutor,
)
from app.services.operation_repository import (
    OperationConflictError,
    OperationEvent,
    OperationRepository,
)
from app.services.discord_receipts import DiscordOperationClaim, DiscordReceiptIdentity
from app.services.resource_lifecycle import claims_are_compatible, normalize_claims


_SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ExecutorGroupSnapshot:
    group_id: str
    executors: tuple[tuple[str, OperationExecutor], ...]


class OperationCoordinator:
    def __init__(
        self,
        repository: OperationRepository,
        *,
        auto_start: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.repository = repository
        self._executors: dict[str, OperationExecutor] = {}
        self._executor_groups: dict[str, set[str]] = {}
        self._cancel_events: dict[str, Event] = {}
        self._active_claims: dict[str, tuple[str, ...]] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()
        self._state_changed = Condition(self._lock)
        self._stop_event = Event()
        self._max_workers = max_workers
        self._pool: ThreadPoolExecutor | None = None
        self._dispatcher: Thread | None = None
        if auto_start:
            self.start()

    def register_executor(self, kind: str, executor: OperationExecutor) -> None:
        if not _SAFE_CODE.fullmatch(kind):
            raise ValueError("operation_kind_invalid")
        with self._lock:
            if any(kind in kinds for kinds in self._executor_groups.values()):
                raise ValueError("operation_kind_owned")
            self._executors[kind] = executor

    def replace_executor_group(
        self,
        group_id: str,
        executors: Mapping[str, OperationExecutor],
    ) -> ExecutorGroupSnapshot:
        if not _SAFE_CODE.fullmatch(group_id):
            raise ValueError("operation_executor_group_invalid")
        if any(not _SAFE_CODE.fullmatch(kind) for kind in executors):
            raise ValueError("operation_kind_invalid")
        with self._lock:
            current_kinds = self._executor_groups.get(group_id, set())
            for kind in executors:
                if kind in self._executors and kind not in current_kinds:
                    raise ValueError("operation_kind_conflict")
            snapshot = ExecutorGroupSnapshot(
                group_id=group_id,
                executors=tuple(
                    (kind, self._executors[kind])
                    for kind in sorted(current_kinds)
                    if kind in self._executors
                ),
            )
            for kind in current_kinds:
                self._executors.pop(kind, None)
            self._executors.update(executors)
            if executors:
                self._executor_groups[group_id] = set(executors)
            else:
                self._executor_groups.pop(group_id, None)
            return snapshot

    def restore_executor_group(self, snapshot: ExecutorGroupSnapshot) -> None:
        with self._lock:
            current_kinds = self._executor_groups.get(snapshot.group_id, set())
            previous = dict(snapshot.executors)
            for kind in previous:
                if kind in self._executors and kind not in current_kinds:
                    raise ValueError("operation_kind_conflict")
            for kind in current_kinds:
                self._executors.pop(kind, None)
            self._executors.update(previous)
            if previous:
                self._executor_groups[snapshot.group_id] = set(previous)
            else:
                self._executor_groups.pop(snapshot.group_id, None)

    def clear_executor_groups(self, *, prefix: str) -> None:
        with self._lock:
            group_ids = tuple(
                group_id
                for group_id in self._executor_groups
                if group_id.startswith(prefix)
            )
            for group_id in group_ids:
                for kind in self._executor_groups.pop(group_id):
                    self._executors.pop(kind, None)

    def start(self) -> None:
        with self._lock:
            if self._dispatcher is not None and self._dispatcher.is_alive():
                return
            self._pool = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="asteria-operation",
            )
            self._stop_event.clear()
            self._dispatcher = Thread(
                target=self._dispatch_loop,
                name="asteria-operation-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()

    def shutdown(self, *, wait: bool = True) -> None:
        self._stop_event.set()
        with self._state_changed:
            self._state_changed.notify_all()
        dispatcher = self._dispatcher
        if dispatcher is not None and dispatcher.is_alive():
            dispatcher.join(timeout=5 if wait else 0)
        pool = self._pool
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=False)
            self._pool = None

    def recover_startup(self) -> list[OperationRecord]:
        return self.repository.interrupt_active_operations()

    def enqueue(
        self,
        *,
        kind: str,
        source_actor: str,
        request_snapshot: dict[str, Any],
        idempotency_key: str | None = None,
        request_contract_version: int = 1,
        progress_total: int | None = None,
        progress_unit: str = "",
        parent_operation_id: str | None = None,
        operation_id: str | None = None,
    ) -> OperationRecord:
        executor = self._executors.get(kind)
        executor_version = executor.version if executor is not None else "unregistered-v1"
        return self.repository.create(
            kind=kind,
            source_actor=source_actor,
            request_snapshot=request_snapshot,
            idempotency_key=idempotency_key,
            request_contract_version=request_contract_version,
            progress_total=progress_total,
            progress_unit=progress_unit,
            parent_operation_id=parent_operation_id,
            executor_version=executor_version,
            operation_id=operation_id,
        )

    def enqueue_with_discord_receipt(
        self,
        *,
        kind: str,
        source_actor: str,
        request_snapshot: dict[str, Any],
        identity: DiscordReceiptIdentity,
        requester_id: str,
        guild_id: str | None,
        channel_id: str | None,
        request_contract_version: int = 1,
        progress_total: int | None = None,
        progress_unit: str = "",
    ):
        executor = self._executors.get(kind)
        executor_version = executor.version if executor is not None else "unregistered-v1"
        operation, created = self.repository.create_with_discord_receipt(
            kind=kind,
            source_actor=source_actor,
            request_snapshot=request_snapshot,
            identity=identity,
            requester_id=requester_id,
            guild_id=guild_id,
            channel_id=channel_id,
            request_contract_version=request_contract_version,
            progress_total=progress_total,
            progress_unit=progress_unit,
            executor_version=executor_version,
        )
        return DiscordOperationClaim(operation=operation, created=created)

    def get(self, operation_id: str) -> OperationRecord:
        return self.repository.get(operation_id)

    def list(self, *, limit: int = 200, for_display: bool = False) -> list[OperationRecord]:
        return self.repository.list(limit=limit, for_display=for_display)

    def events(
        self,
        operation_id: str,
        *,
        after_revision: int = 0,
        limit: int = 200,
    ) -> list[OperationEvent]:
        return self.repository.events(
            operation_id,
            after_revision=after_revision,
            limit=limit,
        )

    def request_cancel(self, operation_id: str) -> OperationRecord:
        while True:
            current = self.repository.get(operation_id)
            if current.status in TERMINAL_OPERATION_STATUSES:
                return current
            target = "cancelled" if current.status == "queued" else "cancelling"
            try:
                updated = self.repository.transition(
                    operation_id,
                    expected_revision=current.row_revision,
                    target_status=target,
                    phase="cancelled" if target == "cancelled" else "cancelling",
                    cancel_requested=True,
                    retryable=target == "cancelled",
                )
                break
            except OperationConflictError:
                continue
        with self._lock:
            event = self._cancel_events.get(operation_id)
            if event is not None:
                event.set()
            self._state_changed.notify_all()
        return updated

    def retry(
        self,
        operation_id: str,
        *,
        source_actor: str,
        idempotency_key: str | None = None,
        request_snapshot: dict[str, Any] | None = None,
    ) -> OperationRecord:
        return self.repository.retry(
            operation_id,
            source_actor=source_actor,
            idempotency_key=idempotency_key,
            request_snapshot=request_snapshot,
        )

    def dismiss(self, operation_id: str) -> OperationRecord:
        while True:
            current = self.repository.get(operation_id)
            try:
                return self.repository.dismiss(
                    operation_id,
                    expected_revision=current.row_revision,
                )
            except OperationConflictError:
                continue

    def execute_next(self) -> OperationRecord | None:
        selected = self._select_ready()
        if selected is None:
            return None
        operation, executor, claims = selected
        cancel_event = Event()
        with self._lock:
            if operation.operation_id in self._active_claims:
                return None
            self._active_claims[operation.operation_id] = claims
            self._cancel_events[operation.operation_id] = cancel_event
        try:
            self._execute(operation.operation_id, executor, cancel_event)
        finally:
            self._release(operation.operation_id)
        return self.repository.get(operation.operation_id)

    def execute_operation(self, operation_id: str) -> OperationRecord:
        operation = self.repository.get(operation_id)
        if operation.status != "queued":
            return operation
        selected = self._select_ready()
        if selected is None or selected[0].operation_id != operation_id:
            return operation
        _selected_operation, executor, claims = selected
        cancel_event = Event()
        with self._lock:
            if operation_id in self._active_claims:
                return self.repository.get(operation_id)
            self._active_claims[operation_id] = claims
            self._cancel_events[operation_id] = cancel_event
        try:
            self._execute(operation_id, executor, cancel_event)
        finally:
            self._release(operation_id)
        return self.repository.get(operation_id)

    def wait_for_terminal(self, operation_id: str) -> OperationRecord:
        """Wait for terminal state or shutdown without racing the dispatcher."""
        with self._state_changed:
            while True:
                operation = self.repository.get(operation_id)
                if operation.status in TERMINAL_OPERATION_STATUSES or self._stop_event.is_set():
                    return operation
                self._state_changed.wait()

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            selected = self._select_ready()
            if selected is None:
                self._stop_event.wait(0.05)
                continue
            operation, executor, claims = selected
            cancel_event = Event()
            with self._lock:
                if operation.operation_id in self._active_claims:
                    continue
                self._active_claims[operation.operation_id] = claims
                self._cancel_events[operation.operation_id] = cancel_event
                pool = self._pool
                if pool is None:
                    return
                future = pool.submit(
                    self._execute_and_release,
                    operation.operation_id,
                    executor,
                    cancel_event,
                )
                self._futures[operation.operation_id] = future
            sleep(0)

    def _select_ready(self) -> tuple[OperationRecord, OperationExecutor, tuple[str, ...]] | None:
        queued = self.repository.list(statuses=("queued",), limit=1000, include_dismissed=True)
        with self._lock:
            active_claim_sets = tuple(self._active_claims.values())
            active_ids = set(self._active_claims)
            executors = dict(self._executors)
        blocked_claim_sets: list[tuple[str, ...]] = []
        for operation in queued:
            if operation.operation_id in active_ids:
                continue
            executor = executors.get(operation.kind)
            if executor is None:
                continue
            claims = normalize_claims(executor.resource_claims(operation.request_snapshot))
            if any(
                not claims_are_compatible(claims, other)
                for other in (*active_claim_sets, *blocked_claim_sets)
            ):
                blocked_claim_sets.append(claims)
                continue
            return operation, executor, claims
        return None

    def _execute_and_release(
        self,
        operation_id: str,
        executor: OperationExecutor,
        cancel_event: Event,
    ) -> None:
        try:
            self._execute(operation_id, executor, cancel_event)
        finally:
            self._release(operation_id)

    def _release(self, operation_id: str) -> None:
        with self._lock:
            self._active_claims.pop(operation_id, None)
            self._cancel_events.pop(operation_id, None)
            self._futures.pop(operation_id, None)
            self._state_changed.notify_all()

    def _transition(self, operation_id: str, target_status: str, **updates: Any) -> OperationRecord:
        while True:
            current = self.repository.get(operation_id)
            if current.status in TERMINAL_OPERATION_STATUSES:
                return current
            try:
                return self.repository.transition(
                    operation_id,
                    expected_revision=current.row_revision,
                    target_status=target_status,
                    **updates,
                )
            except OperationConflictError:
                continue

    def _execute(
        self,
        operation_id: str,
        executor: OperationExecutor,
        cancel_event: Event,
    ) -> None:
        current = self.repository.get(operation_id)
        if current.status != "queued":
            return
        current = self._transition(operation_id, "preparing", phase="preparing")
        current = self._transition(operation_id, "running", phase="running")

        def progress(current_value: int, total: int | None, unit: str, phase: str) -> None:
            self._transition(
                operation_id,
                "running",
                phase=phase,
                progress_current=current_value,
                progress_total=total,
                progress_unit=unit,
            )

        def acknowledge_cancel() -> None:
            latest = self.repository.get(operation_id)
            if latest.status == "cancelling":
                cancel_event.set()

        context = OperationExecutionContext(
            operation=current,
            cancel_event=cancel_event,
            progress_callback=progress,
            cancel_acknowledgement=acknowledge_cancel,
        )
        try:
            result = executor.execute(context)
            if not isinstance(result, OperationExecutionResult):
                raise TypeError("invalid operation result")
            latest = self.repository.get(operation_id)
            terminal = "completed_with_warnings" if result.warning_codes else "completed"
            self._transition(
                operation_id,
                terminal,
                phase=terminal,
                progress_current=latest.progress_total or latest.progress_current,
                artifact_refs=result.artifact_refs,
                warning_codes=result.warning_codes,
                child_summaries=result.child_summaries,
            )
        except OperationCancelled as exc:
            self._transition(
                operation_id,
                "cancelled",
                phase="cancelled",
                retryable=True,
                cancel_requested=True,
                artifact_refs=exc.artifact_refs,
            )
        except OperationExecutionError as exc:
            code = exc.error_code if _SAFE_CODE.fullmatch(exc.error_code) else "operation_executor_failed"
            self._transition(
                operation_id,
                "failed",
                phase="failed",
                error_code=code,
                retryable=exc.retryable,
                artifact_refs=exc.artifact_refs,
            )
        except Exception:
            self._transition(
                operation_id,
                "failed",
                phase="failed",
                error_code="operation_executor_failed",
                retryable=False,
            )
