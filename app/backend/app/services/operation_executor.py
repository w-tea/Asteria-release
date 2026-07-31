from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Callable, Protocol

from app.schemas.operations import OperationChildSummary, OperationRecord


@dataclass(frozen=True, slots=True)
class OperationExecutionResult:
    artifact_refs: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    child_summaries: tuple[OperationChildSummary, ...] = ()


class OperationExecutionError(RuntimeError):
    def __init__(
        self,
        error_code: str,
        *,
        retryable: bool = False,
        artifact_refs: tuple[str, ...] = (),
    ) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable
        self.artifact_refs = artifact_refs


class OperationCancelled(RuntimeError):
    def __init__(self, *, artifact_refs: tuple[str, ...] = ()) -> None:
        super().__init__("operation_cancelled")
        self.artifact_refs = artifact_refs


class OperationExecutionContext:
    def __init__(
        self,
        *,
        operation: OperationRecord,
        cancel_event: Event,
        progress_callback: Callable[[int, int | None, str, str], None],
        cancel_acknowledgement: Callable[[], None],
    ) -> None:
        self.operation = operation
        self.request_snapshot = operation.request_snapshot
        self._cancel_event = cancel_event
        self._progress_callback = progress_callback
        self._cancel_acknowledgement = cancel_acknowledgement

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            self.acknowledge_cancel()
            raise OperationCancelled()

    def acknowledge_cancel(self) -> None:
        self._cancel_acknowledgement()

    def report_progress(
        self,
        current: int,
        total: int | None,
        unit: str,
        *,
        phase: str,
    ) -> None:
        self._progress_callback(current, total, unit, phase)


class OperationExecutor(Protocol):
    version: str

    def resource_claims(self, request_snapshot: dict) -> tuple[str, ...]: ...

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult: ...
