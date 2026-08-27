from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterable
from uuid import uuid4

from app.schemas.operations import (
    ACTIVE_OPERATION_STATUSES,
    TERMINAL_OPERATION_STATUSES,
    OperationChildSummary,
    OperationRecord,
    OperationStatus,
)
from app.services.operation_state import transition_operation


class OperationNotFoundError(LookupError):
    pass


class OperationConflictError(RuntimeError):
    pass


class OperationRetryError(ValueError):
    pass


class OperationDismissError(ValueError):
    pass


class OperationEvent(OperationRecord):
    event_id: int


def canonical_operation_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        json.loads(encoded)
        return encoded
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("operation_snapshot_invalid") from exc


OPERATIONS_TABLE_SQL = """
create table if not exists operations (
    operation_id text primary key,
    kind text not null,
    status text not null,
    created_at text not null,
    queued_at text not null,
    started_at text,
    updated_at text not null,
    completed_at text,
    source_actor text not null,
    request_snapshot_json text not null,
    phase text not null,
    progress_current integer not null,
    progress_total integer,
    progress_unit text not null,
    artifact_refs_json text not null,
    error_code text,
    retryable integer not null,
    retry_of text,
    idempotency_key text unique,
    row_revision integer not null,
    request_contract_version integer not null,
    parent_operation_id text,
    root_operation_id text,
    warning_codes_json text not null,
    cancel_requested_at text,
    last_heartbeat_at text,
    executor_version text not null,
    child_summaries_json text not null,
    dismissed_at text
)
"""

OPERATION_EVENTS_TABLE_SQL = """
create table if not exists operation_events (
    event_id integer primary key autoincrement,
    operation_id text not null,
    row_revision integer not null,
    recorded_at text not null,
    payload_json text not null,
    unique(operation_id, row_revision)
)
"""

OPERATION_ARTIFACT_REFS_TABLE_SQL = """
create table if not exists operation_artifact_refs (
    operation_id text not null,
    artifact_ref text not null,
    recorded_at text not null,
    primary key(operation_id, artifact_ref)
)
"""


def create_operation_tables(connection: sqlite3.Connection) -> None:
    connection.execute(OPERATIONS_TABLE_SQL)
    connection.execute(OPERATION_EVENTS_TABLE_SQL)
    connection.execute(OPERATION_ARTIFACT_REFS_TABLE_SQL)
    connection.execute(
        "create index if not exists operations_status_queue_idx on operations(status, queued_at)"
    )
    connection.execute(
        "create index if not exists operations_parent_idx on operations(parent_operation_id)"
    )
    connection.execute(
        "create index if not exists operation_artifact_ref_idx "
        "on operation_artifact_refs(artifact_ref)"
    )


class OperationRepository:
    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path
        self.clock = clock or (lambda: datetime.now(UTC))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            create_operation_tables(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def create(
        self,
        *,
        kind: str,
        source_actor: str,
        request_snapshot: dict[str, Any],
        idempotency_key: str | None = None,
        request_contract_version: int = 1,
        progress_total: int | None = None,
        progress_unit: str = "",
        retry_of: str | None = None,
        parent_operation_id: str | None = None,
        root_operation_id: str | None = None,
        executor_version: str,
        operation_id: str | None = None,
    ) -> OperationRecord:
        snapshot_json = canonical_operation_json(request_snapshot)
        with self._connect() as connection:
            connection.execute("begin immediate")
            return self._create_in_connection(
                connection,
                kind=kind,
                source_actor=source_actor,
                request_snapshot=request_snapshot,
                snapshot_json=snapshot_json,
                idempotency_key=idempotency_key,
                request_contract_version=request_contract_version,
                progress_total=progress_total,
                progress_unit=progress_unit,
                retry_of=retry_of,
                parent_operation_id=parent_operation_id,
                root_operation_id=root_operation_id,
                executor_version=executor_version,
                operation_id=operation_id,
            )

    def create_with_discord_receipt(
        self,
        *,
        kind: str,
        source_actor: str,
        request_snapshot: dict[str, Any],
        identity: Any,
        requester_id: str,
        guild_id: str | None,
        channel_id: str | None,
        request_contract_version: int,
        progress_total: int | None,
        progress_unit: str,
        executor_version: str,
    ) -> tuple[OperationRecord, bool]:
        from app.services.discord_receipts import create_discord_receipt_tables

        snapshot_json = canonical_operation_json(request_snapshot)
        with self._connect() as connection:
            connection.execute("begin immediate")
            create_discord_receipt_tables(connection)
            existing_receipt = connection.execute(
                """
                select operation_id from discord_interaction_receipts
                where provider = ? and application_id = ? and interaction_id = ?
                """,
                (identity.provider, identity.application_id, identity.interaction_id),
            ).fetchone()
            if existing_receipt is not None:
                existing_operation = connection.execute(
                    "select * from operations where operation_id = ?",
                    (existing_receipt["operation_id"],),
                ).fetchone()
                if existing_operation is None:
                    raise OperationConflictError("discord_receipt_operation_missing")
                return self._row_to_record(existing_operation), False

            record = self._create_in_connection(
                connection,
                kind=kind,
                source_actor=source_actor,
                request_snapshot=request_snapshot,
                snapshot_json=snapshot_json,
                idempotency_key=identity.operation_idempotency_key,
                request_contract_version=request_contract_version,
                progress_total=progress_total,
                progress_unit=progress_unit,
                retry_of=None,
                parent_operation_id=None,
                root_operation_id=None,
                executor_version=executor_version,
                operation_id=None,
            )
            connection.execute(
                """
                insert into discord_interaction_receipts(
                    provider, application_id, interaction_id, operation_id,
                    requester_id, guild_id, channel_id, claimed_at,
                    delivery_status, delivery_error_code
                ) values (?, ?, ?, ?, ?, ?, ?, ?, 'pending', null)
                """,
                (
                    identity.provider,
                    identity.application_id,
                    identity.interaction_id,
                    record.operation_id,
                    requester_id,
                    guild_id,
                    channel_id,
                    record.created_at.isoformat(),
                ),
            )
            return record, True

    def _create_in_connection(
        self,
        connection: sqlite3.Connection,
        *,
        kind: str,
        source_actor: str,
        request_snapshot: dict[str, Any],
        snapshot_json: str,
        idempotency_key: str | None,
        request_contract_version: int,
        progress_total: int | None,
        progress_unit: str,
        retry_of: str | None,
        parent_operation_id: str | None,
        root_operation_id: str | None,
        executor_version: str,
        operation_id: str | None,
    ) -> OperationRecord:
        if idempotency_key is not None:
            existing = connection.execute(
                "select * from operations where idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return self._row_to_record(existing)
        if parent_operation_id is not None:
            parent_row = connection.execute(
                "select * from operations where operation_id = ?",
                (parent_operation_id,),
            ).fetchone()
            if parent_row is None:
                raise OperationNotFoundError("operation_parent_not_found")
            parent = self._row_to_record(parent_row)
            expected_root = parent.root_operation_id or parent.operation_id
            if root_operation_id is not None and root_operation_id != expected_root:
                raise ValueError("operation_root_mismatch")
            root_operation_id = expected_root
        elif root_operation_id is not None:
            raise ValueError("operation_root_without_parent")

        now = self.clock()
        record = OperationRecord(
            operation_id=operation_id or uuid4().hex,
            kind=kind,
            status="queued",
            created_at=now,
            queued_at=now,
            updated_at=now,
            source_actor=source_actor,
            request_snapshot=json.loads(snapshot_json),
            phase="queued",
            progress_current=0,
            progress_total=progress_total,
            progress_unit=progress_unit,
            retry_of=retry_of,
            idempotency_key=idempotency_key,
            row_revision=1,
            request_contract_version=request_contract_version,
            parent_operation_id=parent_operation_id,
            root_operation_id=root_operation_id,
            executor_version=executor_version,
        )
        try:
            connection.execute(
                """
                insert into operations(
                    operation_id, kind, status, created_at, queued_at, started_at, updated_at,
                    completed_at, source_actor, request_snapshot_json, phase, progress_current,
                    progress_total, progress_unit, artifact_refs_json, error_code, retryable,
                    retry_of, idempotency_key, row_revision, request_contract_version,
                    parent_operation_id, root_operation_id, warning_codes_json,
                    cancel_requested_at, last_heartbeat_at, executor_version,
                    child_summaries_json, dismissed_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._record_values(record, snapshot_json=snapshot_json),
            )
        except sqlite3.IntegrityError as exc:
            if idempotency_key is not None:
                existing = connection.execute(
                    "select * from operations where idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    return self._row_to_record(existing)
            raise OperationConflictError("operation_create_conflict") from exc
        self._append_event(connection, record)
        self._replace_artifact_refs(connection, record)
        return record

    def get(self, operation_id: str) -> OperationRecord:
        with self._connect() as connection:
            row = connection.execute(
                "select * from operations where operation_id = ?",
                (operation_id,),
            ).fetchone()
        if row is None:
            raise OperationNotFoundError("operation_not_found")
        return self._row_to_record(row)

    def list(
        self,
        *,
        statuses: Iterable[OperationStatus] | None = None,
        limit: int = 200,
        include_dismissed: bool = False,
    ) -> list[OperationRecord]:
        if limit < 1 or limit > 1000:
            raise ValueError("operation_limit_invalid")
        clauses: list[str] = []
        parameters: list[Any] = []
        if statuses is not None:
            normalized = tuple(dict.fromkeys(statuses))
            if not normalized:
                return []
            clauses.append(f"status in ({','.join('?' for _ in normalized)})")
            parameters.extend(normalized)
        if not include_dismissed:
            clauses.append("dismissed_at is null")
        where = f"where {' and '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"select * from operations {where} order by queued_at asc, rowid asc limit ?",
                (*parameters, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def transition(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        target_status: OperationStatus,
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
        with self._connect() as connection:
            connection.execute("begin immediate")
            row = connection.execute(
                "select * from operations where operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationNotFoundError("operation_not_found")
            current = self._row_to_record(row)
            if current.row_revision != expected_revision:
                raise OperationConflictError("operation_revision_conflict")
            updated = transition_operation(
                current,
                target_status,
                now=self.clock(),
                phase=phase,
                progress_current=progress_current,
                progress_total=progress_total,
                progress_unit=progress_unit,
                artifact_refs=artifact_refs,
                warning_codes=warning_codes,
                error_code=error_code,
                retryable=retryable,
                cancel_requested=cancel_requested,
                last_heartbeat_at=last_heartbeat_at,
                child_summaries=child_summaries,
            )
            cursor = connection.execute(
                """
                update operations set
                    status = ?, started_at = ?, updated_at = ?, completed_at = ?,
                    phase = ?, progress_current = ?, progress_total = ?, progress_unit = ?,
                    artifact_refs_json = ?, error_code = ?, retryable = ?,
                    warning_codes_json = ?, cancel_requested_at = ?, last_heartbeat_at = ?,
                    child_summaries_json = ?, row_revision = ?
                where operation_id = ? and row_revision = ?
                """,
                (
                    updated.status,
                    self._datetime_text(updated.started_at),
                    self._datetime_text(updated.updated_at),
                    self._datetime_text(updated.completed_at),
                    updated.phase,
                    updated.progress_current,
                    updated.progress_total,
                    updated.progress_unit,
                    canonical_operation_json(updated.artifact_refs),
                    updated.error_code,
                    int(updated.retryable),
                    canonical_operation_json(updated.warning_codes),
                    self._datetime_text(updated.cancel_requested_at),
                    self._datetime_text(updated.last_heartbeat_at),
                    canonical_operation_json([
                        item.model_dump(mode="json") for item in updated.child_summaries
                    ]),
                    updated.row_revision,
                    operation_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise OperationConflictError("operation_revision_conflict")
            self._append_event(connection, updated)
            self._replace_artifact_refs(connection, updated)
            return updated

    def append_terminal_child_summary(
        self,
        operation_id: str,
        summary: OperationChildSummary | dict[str, Any],
    ) -> OperationRecord:
        validated = (
            summary
            if isinstance(summary, OperationChildSummary)
            else OperationChildSummary.model_validate(summary)
        )
        if validated.status not in TERMINAL_OPERATION_STATUSES:
            raise ValueError("terminal child summary required")
        with self._connect() as connection:
            connection.execute("begin immediate")
            row = connection.execute(
                "select * from operations where operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationNotFoundError("operation_not_found")
            current = self._row_to_record(row)
            if current.status not in TERMINAL_OPERATION_STATUSES:
                raise OperationConflictError("operation_not_terminal")
            existing = tuple(
                item
                for item in current.child_summaries
                if item.child_summary_id != validated.child_summary_id
            )
            now = self.clock()
            warning_codes = tuple(
                dict.fromkeys((*current.warning_codes, *validated.warning_codes))
            )
            updated = current.model_copy(
                update={
                    "updated_at": now,
                    "warning_codes": warning_codes,
                    "child_summaries": (*existing, validated),
                    "row_revision": current.row_revision + 1,
                }
            ).validate_contract()
            cursor = connection.execute(
                """
                update operations set
                    updated_at = ?, warning_codes_json = ?,
                    child_summaries_json = ?, row_revision = ?
                where operation_id = ? and row_revision = ?
                """,
                (
                    self._datetime_text(updated.updated_at),
                    canonical_operation_json(updated.warning_codes),
                    canonical_operation_json(
                        [item.model_dump(mode="json") for item in updated.child_summaries]
                    ),
                    updated.row_revision,
                    operation_id,
                    current.row_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise OperationConflictError("operation_revision_conflict")
            self._append_event(connection, updated)
            self._replace_artifact_refs(connection, updated)
            return updated

    def retry(
        self,
        operation_id: str,
        *,
        source_actor: str,
        idempotency_key: str | None = None,
        request_snapshot: dict[str, Any] | None = None,
    ) -> OperationRecord:
        original = self.get(operation_id)
        if original.status not in {"failed", "cancelled", "interrupted"} or not original.retryable:
            raise OperationRetryError("operation_retry_not_allowed")
        return self.create(
            kind=original.kind,
            source_actor=source_actor,
            request_snapshot=(
                original.request_snapshot
                if request_snapshot is None
                else request_snapshot
            ),
            idempotency_key=idempotency_key,
            request_contract_version=original.request_contract_version,
            progress_total=original.progress_total,
            progress_unit=original.progress_unit,
            retry_of=original.operation_id,
            parent_operation_id=original.parent_operation_id,
            root_operation_id=original.root_operation_id,
            executor_version=original.executor_version,
        )

    def interrupt_active_operations(self) -> list[OperationRecord]:
        recovered: list[OperationRecord] = []
        for current in self.list(statuses=ACTIVE_OPERATION_STATUSES, limit=1000, include_dismissed=True):
            try:
                recovered.append(self.transition(
                    current.operation_id,
                    expected_revision=current.row_revision,
                    target_status="interrupted",
                    phase="interrupted",
                    error_code="operation_interrupted_by_restart",
                    retryable=True,
                ))
            except OperationConflictError:
                continue
        return recovered

    def dismiss(self, operation_id: str, *, expected_revision: int) -> OperationRecord:
        with self._connect() as connection:
            connection.execute("begin immediate")
            row = connection.execute(
                "select * from operations where operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationNotFoundError("operation_not_found")
            current = self._row_to_record(row)
            if current.row_revision != expected_revision:
                raise OperationConflictError("operation_revision_conflict")
            if current.status not in TERMINAL_OPERATION_STATUSES:
                raise OperationDismissError("operation_dismiss_not_allowed")
            if current.dismissed_at is not None:
                return current
            now = self.clock()
            updated = current.model_copy(update={
                "dismissed_at": now,
                "updated_at": now,
                "row_revision": current.row_revision + 1,
            }).validate_contract()
            cursor = connection.execute(
                """
                update operations
                set dismissed_at = ?, updated_at = ?, row_revision = ?
                where operation_id = ? and row_revision = ?
                """,
                (
                    self._datetime_text(updated.dismissed_at),
                    self._datetime_text(updated.updated_at),
                    updated.row_revision,
                    operation_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise OperationConflictError("operation_revision_conflict")
            self._append_event(connection, updated)
            return updated

    def delete_terminal_metadata(self, operation_ids: Iterable[str]) -> int:
        normalized = tuple(dict.fromkeys(operation_ids))
        if not normalized:
            return 0
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as connection:
            connection.execute("begin immediate")
            rows = connection.execute(
                f"select operation_id, status from operations where operation_id in ({placeholders})",
                normalized,
            ).fetchall()
            deletable = tuple(
                row["operation_id"]
                for row in rows
                if row["status"] in TERMINAL_OPERATION_STATUSES
            )
            if not deletable:
                return 0
            delete_placeholders = ",".join("?" for _ in deletable)
            connection.execute(
                f"delete from operation_artifact_refs where operation_id in ({delete_placeholders})",
                deletable,
            )
            connection.execute(
                f"delete from operation_events where operation_id in ({delete_placeholders})",
                deletable,
            )
            cursor = connection.execute(
                f"delete from operations where operation_id in ({delete_placeholders}) "
                f"and status in ({','.join('?' for _ in TERMINAL_OPERATION_STATUSES)})",
                (*deletable, *TERMINAL_OPERATION_STATUSES),
            )
            return cursor.rowcount

    def events(
        self,
        operation_id: str,
        *,
        after_revision: int = 0,
        limit: int = 200,
    ) -> list[OperationEvent]:
        if after_revision < 0 or limit < 1 or limit > 1000:
            raise ValueError("operation_event_query_invalid")
        self.get(operation_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                select event_id, payload_json
                from operation_events
                where operation_id = ? and row_revision > ?
                order by row_revision asc
                limit ?
                """,
                (operation_id, after_revision, limit),
            ).fetchall()
        return [
            OperationEvent(
                event_id=row["event_id"],
                **json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    @staticmethod
    def _append_event(connection: sqlite3.Connection, record: OperationRecord) -> None:
        connection.execute(
            """
            insert into operation_events(operation_id, row_revision, recorded_at, payload_json)
            values (?, ?, ?, ?)
            """,
            (
                record.operation_id,
                record.row_revision,
                record.updated_at.isoformat(),
                canonical_operation_json(record.model_dump(mode="json")),
            ),
        )

    @staticmethod
    def _replace_artifact_refs(
        connection: sqlite3.Connection,
        record: OperationRecord,
    ) -> None:
        references = {
            *record.artifact_refs,
            *(
                artifact_ref
                for child in record.child_summaries
                for artifact_ref in child.artifact_refs
            ),
        }
        connection.execute(
            "delete from operation_artifact_refs where operation_id = ?",
            (record.operation_id,),
        )
        connection.executemany(
            """
            insert into operation_artifact_refs(operation_id, artifact_ref, recorded_at)
            values (?, ?, ?)
            """,
            (
                (record.operation_id, artifact_ref, record.updated_at.isoformat())
                for artifact_ref in sorted(references)
            ),
        )

    @staticmethod
    def _datetime_text(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def _record_values(
        self,
        record: OperationRecord,
        *,
        snapshot_json: str | None = None,
    ) -> tuple[Any, ...]:
        return (
            record.operation_id,
            record.kind,
            record.status,
            self._datetime_text(record.created_at),
            self._datetime_text(record.queued_at),
            self._datetime_text(record.started_at),
            self._datetime_text(record.updated_at),
            self._datetime_text(record.completed_at),
            record.source_actor,
            snapshot_json or canonical_operation_json(record.request_snapshot),
            record.phase,
            record.progress_current,
            record.progress_total,
            record.progress_unit,
            canonical_operation_json(record.artifact_refs),
            record.error_code,
            int(record.retryable),
            record.retry_of,
            record.idempotency_key,
            record.row_revision,
            record.request_contract_version,
            record.parent_operation_id,
            record.root_operation_id,
            canonical_operation_json(record.warning_codes),
            self._datetime_text(record.cancel_requested_at),
            self._datetime_text(record.last_heartbeat_at),
            record.executor_version,
            canonical_operation_json([item.model_dump(mode="json") for item in record.child_summaries]),
            self._datetime_text(record.dismissed_at),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> OperationRecord:
        return OperationRecord(
            operation_id=row["operation_id"],
            kind=row["kind"],
            status=row["status"],
            created_at=row["created_at"],
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            source_actor=row["source_actor"],
            request_snapshot=json.loads(row["request_snapshot_json"]),
            phase=row["phase"],
            progress_current=row["progress_current"],
            progress_total=row["progress_total"],
            progress_unit=row["progress_unit"],
            artifact_refs=tuple(json.loads(row["artifact_refs_json"])),
            error_code=row["error_code"],
            retryable=bool(row["retryable"]),
            retry_of=row["retry_of"],
            idempotency_key=row["idempotency_key"],
            row_revision=row["row_revision"],
            request_contract_version=row["request_contract_version"],
            parent_operation_id=row["parent_operation_id"],
            root_operation_id=row["root_operation_id"],
            warning_codes=tuple(json.loads(row["warning_codes_json"])),
            cancel_requested_at=row["cancel_requested_at"],
            last_heartbeat_at=row["last_heartbeat_at"],
            executor_version=row["executor_version"],
            child_summaries=tuple(json.loads(row["child_summaries_json"])),
            dismissed_at=row["dismissed_at"],
        )
