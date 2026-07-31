from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


OperationStatus = Literal[
    "queued",
    "preparing",
    "running",
    "cancelling",
    "completed",
    "completed_with_warnings",
    "failed",
    "cancelled",
    "interrupted",
]

ACTIVE_OPERATION_STATUSES = frozenset({"queued", "preparing", "running", "cancelling"})
TERMINAL_OPERATION_STATUSES = frozenset(
    {"completed", "completed_with_warnings", "failed", "cancelled", "interrupted"}
)
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _validate_token(value: str, label: str, *, allow_empty: bool = False) -> None:
    if allow_empty and not value:
        return
    if not _TOKEN.fullmatch(value):
        raise ValueError(f"{label} is invalid")


def _validate_timestamp(value: datetime | None, label: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{label} must include timezone")


class OperationChildSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    child_summary_id: str
    ordinal: int = Field(ge=0)
    kind: str
    status: OperationStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    artifact_refs: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> "OperationChildSummary":
        _validate_token(self.child_summary_id, "child summary id")
        _validate_token(self.kind, "child summary kind")
        _validate_timestamp(self.started_at, "child summary started_at")
        _validate_timestamp(self.completed_at, "child summary completed_at")
        for value in self.artifact_refs:
            _validate_token(value, "artifact ref")
        for value in self.warning_codes:
            _validate_token(value, "warning code")
        if self.error_code is not None:
            _validate_token(self.error_code, "error code")
        if self.status in TERMINAL_OPERATION_STATUSES and self.completed_at is None:
            raise ValueError("terminal child summary requires completed_at")
        return self


class OperationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_id: str
    kind: str
    status: OperationStatus
    created_at: datetime
    queued_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    completed_at: datetime | None = None
    source_actor: str
    request_snapshot: dict[str, Any]
    phase: str
    progress_current: int = Field(ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    progress_unit: str
    artifact_refs: tuple[str, ...] = ()
    error_code: str | None = None
    retryable: bool = False
    retry_of: str | None = None
    idempotency_key: str | None = None
    row_revision: int = Field(ge=1)
    request_contract_version: int = Field(ge=1)
    parent_operation_id: str | None = None
    root_operation_id: str | None = None
    warning_codes: tuple[str, ...] = ()
    cancel_requested_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    executor_version: str
    child_summaries: tuple[OperationChildSummary, ...] = ()
    dismissed_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "OperationRecord":
        _validate_token(self.operation_id, "operation id")
        _validate_token(self.kind, "operation kind")
        _validate_token(self.source_actor, "source actor")
        _validate_token(self.phase, "operation phase")
        _validate_token(self.progress_unit, "progress unit", allow_empty=True)
        _validate_token(self.executor_version, "executor version")
        for label, value in (
            ("retry_of", self.retry_of),
            ("idempotency_key", self.idempotency_key),
            ("parent_operation_id", self.parent_operation_id),
            ("root_operation_id", self.root_operation_id),
            ("error_code", self.error_code),
        ):
            if value is not None:
                _validate_token(value, label)
        for value in self.artifact_refs:
            _validate_token(value, "artifact ref")
        for value in self.warning_codes:
            _validate_token(value, "warning code")
        for label, value in (
            ("created_at", self.created_at),
            ("queued_at", self.queued_at),
            ("started_at", self.started_at),
            ("updated_at", self.updated_at),
            ("completed_at", self.completed_at),
            ("cancel_requested_at", self.cancel_requested_at),
            ("last_heartbeat_at", self.last_heartbeat_at),
            ("dismissed_at", self.dismissed_at),
        ):
            _validate_timestamp(value, label)
        if self.progress_total is not None and self.progress_current > self.progress_total:
            raise ValueError("operation progress is invalid")
        if self.status in TERMINAL_OPERATION_STATUSES and self.completed_at is None:
            raise ValueError("terminal operation requires completed_at")
        if self.status not in TERMINAL_OPERATION_STATUSES and self.completed_at is not None:
            raise ValueError("active operation cannot have completed_at")
        if self.parent_operation_id is None and self.root_operation_id is not None:
            raise ValueError("root operation requires parent operation")
        pairs = [(item.child_summary_id, item.ordinal) for item in self.child_summaries]
        if len(pairs) != len(set(pairs)):
            raise ValueError("duplicate child summary")
        return self

    def validate_contract(self) -> "OperationRecord":
        return type(self).model_validate(self.model_dump())
