from __future__ import annotations

from datetime import datetime
from typing import Any

from app.schemas.operations import (
    TERMINAL_OPERATION_STATUSES,
    OperationChildSummary,
    OperationRecord,
    OperationStatus,
)


class OperationTransitionError(ValueError):
    pass


_TRANSITIONS: dict[OperationStatus, frozenset[OperationStatus]] = {
    "queued": frozenset({"queued", "preparing", "running", "cancelling", "cancelled", "failed", "interrupted"}),
    "preparing": frozenset({"preparing", "running", "cancelling", "cancelled", "failed", "interrupted"}),
    "running": frozenset(
        {"running", "cancelling", "completed", "completed_with_warnings", "failed", "cancelled", "interrupted"}
    ),
    "cancelling": frozenset(
        {"cancelling", "completed", "completed_with_warnings", "failed", "cancelled", "interrupted"}
    ),
    "completed": frozenset(),
    "completed_with_warnings": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}


def transition_operation(
    record: OperationRecord,
    target_status: OperationStatus,
    *,
    now: datetime,
    phase: str | None = None,
    progress_current: int | None = None,
    progress_total: int | None = None,
    progress_unit: str | None = None,
    artifact_refs: tuple[str, ...] | None = None,
    warning_codes: tuple[str, ...] | None = None,
    error_code: str | None = None,
    retryable: bool | None = None,
    cancel_requested: bool = False,
    last_heartbeat_at: datetime | None = None,
    child_summaries: tuple[OperationChildSummary | dict[str, Any], ...] | None = None,
) -> OperationRecord:
    if record.status in TERMINAL_OPERATION_STATUSES:
        raise OperationTransitionError("operation_terminal")
    if target_status not in _TRANSITIONS[record.status]:
        raise OperationTransitionError("operation_transition_invalid")
    if now.tzinfo is None:
        raise OperationTransitionError("operation_timestamp_invalid")

    next_progress_current = record.progress_current if progress_current is None else progress_current
    next_progress_total = record.progress_total if progress_total is None else progress_total
    if next_progress_current < 0 or (
        next_progress_total is not None
        and (next_progress_total < 0 or next_progress_current > next_progress_total)
    ):
        raise OperationTransitionError("operation_progress_invalid")

    started_at = record.started_at
    if target_status == "running" and started_at is None:
        started_at = now
    completed_at = record.completed_at
    if target_status in TERMINAL_OPERATION_STATUSES and completed_at is None:
        completed_at = now
    cancel_requested_at = record.cancel_requested_at
    if cancel_requested and cancel_requested_at is None:
        cancel_requested_at = now
    summaries = record.child_summaries
    if child_summaries is not None:
        summaries = tuple(
            item if isinstance(item, OperationChildSummary) else OperationChildSummary.model_validate(item)
            for item in child_summaries
        )

    try:
        return record.model_copy(update={
            "status": target_status,
            "started_at": started_at,
            "updated_at": now,
            "completed_at": completed_at,
            "phase": phase if phase is not None else record.phase,
            "progress_current": next_progress_current,
            "progress_total": next_progress_total,
            "progress_unit": progress_unit if progress_unit is not None else record.progress_unit,
            "artifact_refs": artifact_refs if artifact_refs is not None else record.artifact_refs,
            "warning_codes": warning_codes if warning_codes is not None else record.warning_codes,
            "error_code": error_code,
            "retryable": retryable if retryable is not None else record.retryable,
            "cancel_requested_at": cancel_requested_at,
            "last_heartbeat_at": last_heartbeat_at if last_heartbeat_at is not None else record.last_heartbeat_at,
            "child_summaries": summaries,
            "row_revision": record.row_revision + 1,
        }).validate_contract()
    except ValueError as exc:
        code = str(exc)
        if "progress" in code:
            raise OperationTransitionError("operation_progress_invalid") from exc
        raise OperationTransitionError("operation_contract_invalid") from exc
