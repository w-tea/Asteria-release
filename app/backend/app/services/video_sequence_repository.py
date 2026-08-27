from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas.video import VideoRequest
from app.schemas.video_sequence import VideoSequence, VideoSequenceSegment


class VideoSequenceRepositoryError(RuntimeError):
    code = "video_sequence_repository_error"


class VideoSequenceNotFoundError(VideoSequenceRepositoryError):
    code = "video_sequence_not_found"


class VideoSequenceConflictError(VideoSequenceRepositoryError):
    code = "video_sequence_revision_conflict"


class VideoSequenceTransitionError(VideoSequenceRepositoryError):
    code = "video_sequence_transition_invalid"


def create_video_sequence_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        create table if not exists video_sequences (
            sequence_id text primary key,
            document_json text not null,
            row_revision integer not null,
            status text not null,
            active_chain_revision integer not null,
            created_at text not null,
            updated_at text not null
        )
        """
    )
    connection.execute(
        "create index if not exists idx_video_sequences_updated_at "
        "on video_sequences(updated_at desc)"
    )
    connection.execute(
        """
        create table if not exists video_sequence_transition_receipts (
            receipt_id text primary key,
            sequence_id text not null,
            action text not null,
            request_hash text not null,
            sequence_revision integer not null,
            result_json text not null,
            created_at text not null,
            foreign key(sequence_id) references video_sequences(sequence_id) on delete cascade
        )
        """
    )
    connection.execute(
        "create index if not exists idx_video_sequence_receipts_sequence "
        "on video_sequence_transition_receipts(sequence_id, sequence_revision)"
    )
    connection.execute(
        """
        create table if not exists video_sequence_delete_receipts (
            receipt_id text primary key,
            sequence_id text not null,
            request_hash text not null,
            created_at text not null
        )
        """
    )
    connection.execute(
        """
        create table if not exists video_sequence_action_receipts (
            receipt_id text primary key,
            sequence_id text not null,
            action text not null,
            request_hash text not null,
            created_at text not null
        )
        """
    )


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_payload(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _request_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


class VideoSequenceRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path.resolve()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            create_video_sequence_tables(connection)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma foreign_keys = on")
        return connection

    def get(self, sequence_id: str) -> VideoSequence:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "select document_json from video_sequences where sequence_id = ?",
                (sequence_id,),
            ).fetchone()
        if row is None:
            raise VideoSequenceNotFoundError("video sequence was not found")
        return self._decode(str(row["document_json"]))

    def list(self, *, limit: int = 100) -> list[VideoSequence]:
        bounded = max(1, min(limit, 1000))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "select document_json from video_sequences order by updated_at desc limit ?",
                (bounded,),
            ).fetchall()
        return [self._decode(str(row["document_json"])) for row in rows]

    def create(self, sequence: VideoSequence, *, receipt_id: str) -> VideoSequence:
        request_payload = sequence.model_dump(
            mode="json",
            exclude={"created_at", "updated_at"},
        )
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            replay = self._receipt_replay(
                connection,
                receipt_id=receipt_id,
                sequence_id=sequence.sequence_id,
                action="create",
                request_payload=request_payload,
            )
            if replay is not None:
                connection.commit()
                return replay
            existing = connection.execute(
                "select 1 from video_sequences where sequence_id = ?",
                (sequence.sequence_id,),
            ).fetchone()
            if existing is not None:
                raise VideoSequenceConflictError("video sequence already exists")
            result = sequence
            self._insert_sequence(connection, result)
            self._store_receipt(
                connection,
                receipt_id=receipt_id,
                sequence_id=result.sequence_id,
                action="create",
                request_payload=request_payload,
                result=result,
            )
            connection.commit()
            return result

    def start_generation(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        operation_id: str | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            if index != len(current.active_segment_ids):
                raise VideoSequenceTransitionError(
                    "segment ordinal is not the next active-chain position"
                )
            if segment.status == "needs_regeneration":
                raise VideoSequenceTransitionError(
                    "later ordinal must be prepared against the new parent revision before generation"
                )
            if segment.status not in {"draft", "retryable"}:
                raise VideoSequenceTransitionError("video sequence segment is not ready for generation")
            updated = self._updated_segment(
                segment,
                status="generating",
                operation_id=operation_id,
            )
            return self._replace_segment(current, index, updated, status="generating")

        return self._mutate(
            sequence_id,
            action="start_generation",
            request_payload={
                "segment_id": segment_id,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
                "operation_id": operation_id,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def enter_review(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        candidate_artifact_id: str,
        candidate_generation_id: int | None = None,
        context_receipt: dict[str, Any] | None,
        working_artifact_receipt: dict[str, Any] | None = None,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        operation_id: str | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            if segment.status != "generating":
                raise VideoSequenceTransitionError("only a generating segment can enter review")
            if operation_id is not None and segment.operation_id != operation_id:
                raise VideoSequenceConflictError("video sequence operation is stale")
            candidates = segment.candidate_artifact_ids
            if candidate_artifact_id not in candidates:
                candidates = (*candidates, candidate_artifact_id)
            candidate_generations = dict(segment.candidate_generation_ids)
            if candidate_generation_id is not None:
                candidate_generations[candidate_artifact_id] = candidate_generation_id
            candidate_receipts = dict(segment.candidate_context_receipts)
            if context_receipt is not None:
                candidate_receipts[candidate_artifact_id] = context_receipt
            candidate_working = dict(segment.candidate_working_artifacts)
            if working_artifact_receipt is not None:
                candidate_working[candidate_artifact_id] = working_artifact_receipt
            updated = self._updated_segment(
                segment,
                candidate_artifact_ids=candidates,
                candidate_generation_ids=candidate_generations,
                candidate_context_receipts=candidate_receipts,
                candidate_working_artifacts=candidate_working,
                context_receipt=context_receipt,
                status="review",
            )
            return self._replace_segment(current, index, updated, status="review")

        return self._mutate(
            sequence_id,
            action="enter_review",
            request_payload={
                "segment_id": segment_id,
                "candidate_artifact_id": candidate_artifact_id,
                "candidate_generation_id": candidate_generation_id,
                "working_artifact_receipt": working_artifact_receipt,
                "context_receipt": context_receipt,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
                "operation_id": operation_id,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def retry_segment(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        operation_id: str | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            if segment.status not in {"review", "retryable", "failed"}:
                raise VideoSequenceTransitionError("video sequence segment cannot be retried")
            updated = self._updated_segment(
                segment,
                status="generating",
                operation_id=operation_id,
            )
            return self._replace_segment(current, index, updated, status="generating")

        return self._mutate(
            sequence_id,
            action="retry_segment",
            request_payload={
                "segment_id": segment_id,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
                "operation_id": operation_id,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def accept_and_continue(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        candidate_artifact_id: str,
        next_segment_id: str,
        next_request_snapshot: VideoRequest,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            accepted = self._accept_review_candidate(segment, candidate_artifact_id)
            if index != len(current.active_segment_ids):
                raise VideoSequenceTransitionError("only the next reviewed segment can extend the chain")
            if index + 1 < len(current.segments):
                next_segment = current.segments[index + 1]
                if next_segment.segment_id != next_segment_id:
                    raise VideoSequenceTransitionError(
                        "historical successor identifier does not match the next ordinal"
                    )
                segments = (*current.segments[:index], accepted, *current.segments[index + 1 :])
            else:
                next_segment = VideoSequenceSegment(
                    segment_id=next_segment_id,
                    ordinal=index + 1,
                    parent_segment_revision=accepted.accepted_revision,
                    request_snapshot=next_request_snapshot,
                    status="draft",
                )
                segments = (*current.segments[:index], accepted, next_segment)
            return self._updated_sequence(
                current,
                segments=segments,
                active_segment_ids=(*current.active_segment_ids, segment_id),
                status="ready_to_continue",
            )

        return self._mutate(
            sequence_id,
            action="accept_and_continue",
            request_payload={
                "segment_id": segment_id,
                "candidate_artifact_id": candidate_artifact_id,
                "next_segment_id": next_segment_id,
                "next_request_snapshot": next_request_snapshot.model_dump(mode="json"),
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def accept_and_finish(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        candidate_artifact_id: str,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        assembly_operation_id: str | None = None,
        assembly_generation_id: int | None = None,
        assembly_revision: int = 1,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            accepted = self._accept_review_candidate(segment, candidate_artifact_id)
            if index != len(current.active_segment_ids):
                raise VideoSequenceTransitionError("only the next reviewed segment can finish the chain")
            segments = list(current.segments)
            segments[index] = accepted
            return self._updated_sequence(
                current,
                segments=tuple(segments),
                active_segment_ids=(*current.active_segment_ids, segment_id),
                assembly_operation_id=assembly_operation_id,
                assembly_generation_id=assembly_generation_id,
                assembly_revision=assembly_revision if assembly_operation_id is not None else 0,
                assembled_artifact_id=None,
                status="assembling",
            )

        return self._mutate(
            sequence_id,
            action="accept_and_finish",
            request_payload={
                "segment_id": segment_id,
                "candidate_artifact_id": candidate_artifact_id,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
                "assembly_operation_id": assembly_operation_id,
                "assembly_generation_id": assembly_generation_id,
                "assembly_revision": assembly_revision,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def start_assembly_retry(
        self,
        sequence_id: str,
        *,
        previous_operation_id: str,
        operation_id: str,
        expected_revision: int,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            if current.status != "failed" or current.assembly_operation_id != previous_operation_id:
                raise VideoSequenceTransitionError("video sequence assembly is not retryable")
            return self._updated_sequence(
                current,
                assembly_operation_id=operation_id,
                status="assembling",
            )

        return self._mutate(
            sequence_id,
            action="start_assembly_retry",
            request_payload={
                "previous_operation_id": previous_operation_id,
                "operation_id": operation_id,
                "expected_revision": expected_revision,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def attach_assembly_generation(
        self,
        sequence_id: str,
        *,
        operation_id: str,
        generation_id: int,
    ) -> VideoSequence:
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            current = self._load_for_update(connection, sequence_id)
            if current.assembly_operation_id != operation_id:
                raise VideoSequenceConflictError("video sequence assembly operation is stale")
            if current.assembly_generation_id is not None:
                if current.assembly_generation_id != generation_id:
                    raise VideoSequenceConflictError("video sequence assembly generation is stale")
                connection.commit()
                return current
            result = self._updated_sequence(current, assembly_generation_id=generation_id)
            self._update_sequence(connection, current, result)
            connection.commit()
            return result

    def complete_assembly(
        self,
        sequence_id: str,
        *,
        operation_id: str,
        assembled_artifact_id: str,
        expected_active_chain_revision: int,
        expected_assembly_revision: int,
    ) -> VideoSequence:
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            current = self._load_for_update(connection, sequence_id)
            if current.assembly_operation_id != operation_id:
                raise VideoSequenceConflictError("video sequence assembly operation is stale")
            if current.status != "assembling":
                raise VideoSequenceTransitionError("video sequence is not assembling")
            if (
                current.active_chain_revision != expected_active_chain_revision
                or current.assembly_revision != expected_assembly_revision
            ):
                raise VideoSequenceConflictError("video sequence assembly snapshot is stale")
            result = self._updated_sequence(
                current,
                assembled_artifact_id=assembled_artifact_id,
                status="completed",
            )
            self._update_sequence(connection, current, result)
            connection.commit()
            return result

    def mark_assembly_failed(
        self,
        sequence_id: str,
        *,
        operation_id: str,
    ) -> VideoSequence:
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            current = self._load_for_update(connection, sequence_id)
            if current.assembly_operation_id != operation_id or current.status not in {
                "assembling",
                "completed",
            }:
                connection.commit()
                return current
            result = self._updated_sequence(
                current,
                assembled_artifact_id=None,
                status="failed",
            )
            self._update_sequence(connection, current, result)
            connection.commit()
            return result

    def add_candidate_history(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        candidate_artifact_id: str,
        context_receipt: dict[str, Any],
        expected_revision: int,
        expected_segment_revision: int | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            candidates = segment.candidate_artifact_ids
            if candidate_artifact_id not in candidates:
                candidates = (*candidates, candidate_artifact_id)
            candidate_receipts = dict(segment.candidate_context_receipts)
            candidate_receipts[candidate_artifact_id] = context_receipt
            updated = self._updated_segment(
                segment,
                candidate_artifact_ids=candidates,
                candidate_context_receipts=candidate_receipts,
                context_receipt=context_receipt,
            )
            return self._replace_segment(current, index, updated, status=current.status)

        return self._mutate(
            sequence_id,
            action="add_candidate_history",
            request_payload={
                "segment_id": segment_id,
                "candidate_artifact_id": candidate_artifact_id,
                "context_receipt": context_receipt,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def replace_accepted(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        candidate_artifact_id: str,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            if segment.accepted_candidate_id is None or segment.accepted_revision is None:
                raise VideoSequenceTransitionError("video sequence segment has no accepted candidate")
            if candidate_artifact_id not in segment.candidate_artifact_ids:
                raise VideoSequenceTransitionError("replacement candidate is not in segment history")
            replacement = self._updated_segment(
                segment,
                accepted_candidate_id=candidate_artifact_id,
                accepted_revision=segment.accepted_revision + 1,
                context_receipt=segment.candidate_context_receipts.get(candidate_artifact_id),
                status="accepted",
            )
            segments = list(current.segments)
            segments[index] = replacement
            for later_index in range(index + 1, len(segments)):
                segments[later_index] = self._updated_segment(
                    segments[later_index],
                    status="needs_regeneration",
                )
            return self._updated_sequence(
                current,
                segments=tuple(segments),
                active_segment_ids=tuple(item.segment_id for item in segments[: index + 1]),
                active_chain_revision=current.active_chain_revision + 1,
                assembly_operation_id=None,
                assembly_generation_id=None,
                assembled_artifact_id=None,
                status="ready_to_continue",
            )

        return self._mutate(
            sequence_id,
            action="replace_accepted",
            request_payload={
                "segment_id": segment_id,
                "candidate_artifact_id": candidate_artifact_id,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def prepare_regeneration(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        request_snapshot: VideoRequest,
        parent_segment_revision: int,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            if segment.status != "needs_regeneration" or index == 0:
                raise VideoSequenceTransitionError("segment ordinal is not awaiting regeneration")
            predecessor = current.segments[index - 1]
            if (
                predecessor.segment_id not in current.active_segment_ids
                or predecessor.accepted_revision != parent_segment_revision
            ):
                raise VideoSequenceTransitionError("segment parent revision is stale")
            updated = self._updated_segment(
                segment,
                parent_segment_revision=parent_segment_revision,
                request_snapshot=request_snapshot,
                context_receipt=None,
                status="draft",
            )
            return self._replace_segment(current, index, updated, status="ready_to_continue")

        return self._mutate(
            sequence_id,
            action="prepare_regeneration",
            request_payload={
                "segment_id": segment_id,
                "request_snapshot": request_snapshot.model_dump(mode="json"),
                "parent_segment_revision": parent_segment_revision,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def resume(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        expected_revision: int,
        expected_segment_revision: int | None = None,
        receipt_id: str,
    ) -> VideoSequence:
        def change(current: VideoSequence) -> VideoSequence:
            _index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            if current.status != "interrupted" or segment.status != "retryable":
                raise VideoSequenceTransitionError("video sequence is not interrupted")
            return self._updated_sequence(current, status="ready_to_continue")

        return self._mutate(
            sequence_id,
            action="resume",
            request_payload={
                "segment_id": segment_id,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
            },
            expected_revision=expected_revision,
            receipt_id=receipt_id,
            change=change,
        )

    def mark_operation_retryable(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        operation_id: str,
        transition_receipt_id: str | None = None,
    ) -> VideoSequence:
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            current = self._load_for_update(connection, sequence_id)
            index, segment = self._segment(current, segment_id)
            if segment.operation_id != operation_id:
                connection.commit()
                return current
            if segment.status != "generating":
                connection.commit()
                return current
            updated = self._updated_segment(segment, status="retryable")
            result = self._replace_segment(current, index, updated, status="interrupted")
            self._update_sequence(connection, current, result)
            if transition_receipt_id is not None:
                connection.execute(
                    "delete from video_sequence_transition_receipts "
                    "where receipt_id = ? and sequence_id = ? "
                    "and action in ('start_generation', 'retry_segment')",
                    (transition_receipt_id, sequence_id),
                )
            connection.commit()
            return result

    def delete_draft(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        expected_revision: int,
        expected_segment_revision: int,
        receipt_id: str,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            request_payload = {
                "sequence_id": sequence_id,
                "segment_id": segment_id,
                "expected_revision": expected_revision,
                "expected_segment_revision": expected_segment_revision,
            }
            request_digest = _request_hash(request_payload)
            replay = connection.execute(
                "select sequence_id, request_hash from video_sequence_delete_receipts "
                "where receipt_id = ?",
                (receipt_id,),
            ).fetchone()
            if replay is not None:
                if (
                    str(replay["sequence_id"]) != sequence_id
                    or str(replay["request_hash"]) != request_digest
                ):
                    raise VideoSequenceConflictError(
                        "video sequence receipt was reused for another request"
                    )
                connection.commit()
                return
            current = self._load_for_update(connection, sequence_id)
            if current.revision != expected_revision:
                raise VideoSequenceConflictError("video sequence revision is stale")
            _index, segment = self._segment(current, segment_id)
            self._require_segment_revision(segment, expected_segment_revision)
            if (
                current.status != "draft"
                or len(current.segments) != 1
                or segment.status != "draft"
                or segment.candidate_artifact_ids
                or segment.operation_id is not None
            ):
                raise VideoSequenceTransitionError("only an unused draft sequence can be deleted")
            connection.execute(
                "delete from video_sequences where sequence_id = ? and row_revision = ?",
                (sequence_id, current.revision),
            )
            connection.execute(
                "insert into video_sequence_delete_receipts("
                "receipt_id, sequence_id, request_hash, created_at) values (?, ?, ?, ?)",
                (receipt_id, sequence_id, request_digest, _now().isoformat()),
            )
            connection.commit()

    def action_receipt_replay(
        self,
        *,
        receipt_id: str,
        sequence_id: str,
        action: str,
        request_payload: dict[str, Any],
    ) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "select sequence_id, action, request_hash "
                "from video_sequence_action_receipts where receipt_id = ?",
                (receipt_id,),
            ).fetchone()
        if row is None:
            return False
        if (
            str(row["sequence_id"]) != sequence_id
            or str(row["action"]) != action
            or str(row["request_hash"]) != _request_hash(request_payload)
        ):
            raise VideoSequenceConflictError(
                "video sequence receipt was reused for another request"
            )
        return True

    def store_action_receipt(
        self,
        *,
        receipt_id: str,
        sequence_id: str,
        action: str,
        request_payload: dict[str, Any],
    ) -> None:
        request_digest = _request_hash(request_payload)
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            row = connection.execute(
                "select sequence_id, action, request_hash "
                "from video_sequence_action_receipts where receipt_id = ?",
                (receipt_id,),
            ).fetchone()
            if row is not None:
                if (
                    str(row["sequence_id"]) != sequence_id
                    or str(row["action"]) != action
                    or str(row["request_hash"]) != request_digest
                ):
                    raise VideoSequenceConflictError(
                        "video sequence receipt was reused for another request"
                    )
                connection.commit()
                return
            connection.execute(
                "insert into video_sequence_action_receipts("
                "receipt_id, sequence_id, action, request_hash, created_at) "
                "values (?, ?, ?, ?, ?)",
                (receipt_id, sequence_id, action, request_digest, _now().isoformat()),
            )
            connection.commit()

    def interrupt_active_sequences(self) -> int:
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            rows = connection.execute(
                "select document_json from video_sequences "
                "where status in ('generating', 'assembling')"
            ).fetchall()
            changed = 0
            for row in rows:
                current = self._decode(str(row["document_json"]))
                if current.status == "assembling" and current.assembly_operation_id is not None:
                    result = self._updated_sequence(current, status="failed")
                elif current.status == "assembling":
                    continue
                else:
                    segments = tuple(
                        self._updated_segment(segment, status="retryable")
                        if segment.status == "generating"
                        else segment
                        for segment in current.segments
                    )
                    result = self._updated_sequence(current, segments=segments, status="interrupted")
                self._update_sequence(connection, current, result)
                changed += 1
            connection.commit()
            return changed

    def _mutate(
        self,
        sequence_id: str,
        *,
        action: str,
        request_payload: dict[str, Any],
        expected_revision: int,
        receipt_id: str,
        change: Callable[[VideoSequence], VideoSequence],
    ) -> VideoSequence:
        with closing(self._connect()) as connection:
            connection.execute("begin immediate")
            replay = self._receipt_replay(
                connection,
                receipt_id=receipt_id,
                sequence_id=sequence_id,
                action=action,
                request_payload=request_payload,
            )
            if replay is not None:
                connection.commit()
                return replay
            current = self._load_for_update(connection, sequence_id)
            if current.revision != expected_revision:
                raise VideoSequenceConflictError("video sequence revision is stale")
            result = change(current)
            self._update_sequence(connection, current, result)
            self._store_receipt(
                connection,
                receipt_id=receipt_id,
                sequence_id=sequence_id,
                action=action,
                request_payload=request_payload,
                result=result,
            )
            connection.commit()
            return result

    def _receipt_replay(
        self,
        connection: sqlite3.Connection,
        *,
        receipt_id: str,
        sequence_id: str,
        action: str,
        request_payload: Any,
    ) -> VideoSequence | None:
        row = connection.execute(
            "select sequence_id, action, request_hash, result_json "
            "from video_sequence_transition_receipts where receipt_id = ?",
            (receipt_id,),
        ).fetchone()
        if row is None:
            return None
        if (
            str(row["sequence_id"]) != sequence_id
            or str(row["action"]) != action
            or str(row["request_hash"]) != _request_hash(request_payload)
        ):
            raise VideoSequenceConflictError("video sequence receipt was reused for another request")
        return self._decode(str(row["result_json"]))

    def _store_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        receipt_id: str,
        sequence_id: str,
        action: str,
        request_payload: Any,
        result: VideoSequence,
    ) -> None:
        connection.execute(
            """
            insert into video_sequence_transition_receipts(
                receipt_id, sequence_id, action, request_hash,
                sequence_revision, result_json, created_at
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                sequence_id,
                action,
                _request_hash(request_payload),
                result.revision,
                self._encode(result),
                _now().isoformat(),
            ),
        )

    def _load_for_update(
        self, connection: sqlite3.Connection, sequence_id: str
    ) -> VideoSequence:
        row = connection.execute(
            "select document_json from video_sequences where sequence_id = ?",
            (sequence_id,),
        ).fetchone()
        if row is None:
            raise VideoSequenceNotFoundError("video sequence was not found")
        return self._decode(str(row["document_json"]))

    @staticmethod
    def _segment(current: VideoSequence, segment_id: str) -> tuple[int, VideoSequenceSegment]:
        for index, segment in enumerate(current.segments):
            if segment.segment_id == segment_id:
                return index, segment
        raise VideoSequenceNotFoundError("video sequence segment was not found")

    @staticmethod
    def _accept_review_candidate(
        segment: VideoSequenceSegment, candidate_artifact_id: str
    ) -> VideoSequenceSegment:
        if segment.status != "review":
            raise VideoSequenceTransitionError("segment must be in review before it can be accepted")
        if candidate_artifact_id not in segment.candidate_artifact_ids:
            raise VideoSequenceTransitionError("accepted candidate is not in segment history")
        return VideoSequenceRepository._updated_segment(
            segment,
            accepted_candidate_id=candidate_artifact_id,
            accepted_revision=(segment.accepted_revision or 0) + 1,
            status="accepted",
        )

    @staticmethod
    def _updated_segment(
        segment: VideoSequenceSegment,
        **updates: Any,
    ) -> VideoSequenceSegment:
        try:
            return VideoSequenceSegment.model_validate(
                {
                    **segment.model_dump(mode="json"),
                    "revision": segment.revision + 1,
                    **updates,
                }
            )
        except ValidationError as exc:
            raise VideoSequenceTransitionError("video sequence segment update is invalid") from exc

    @staticmethod
    def _require_segment_revision(
        segment: VideoSequenceSegment,
        expected_segment_revision: int | None,
    ) -> None:
        if (
            expected_segment_revision is not None
            and segment.revision != expected_segment_revision
        ):
            raise VideoSequenceConflictError("video sequence segment revision is stale")

    def _replace_segment(
        self,
        current: VideoSequence,
        index: int,
        segment: VideoSequenceSegment,
        *,
        status: str,
    ) -> VideoSequence:
        segments = list(current.segments)
        segments[index] = segment
        return self._updated_sequence(current, segments=tuple(segments), status=status)

    @staticmethod
    def _updated_sequence(current: VideoSequence, **updates: Any) -> VideoSequence:
        return VideoSequence.model_validate(
            {
                **current.model_dump(mode="json"),
                **updates,
                "revision": current.revision + 1,
                "updated_at": _now(),
            }
        )

    def _insert_sequence(self, connection: sqlite3.Connection, value: VideoSequence) -> None:
        connection.execute(
            """
            insert into video_sequences(
                sequence_id, document_json, row_revision, status,
                active_chain_revision, created_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.sequence_id,
                self._encode(value),
                value.revision,
                value.status,
                value.active_chain_revision,
                value.created_at.isoformat(),
                value.updated_at.isoformat(),
            ),
        )

    def _update_sequence(
        self,
        connection: sqlite3.Connection,
        previous: VideoSequence,
        value: VideoSequence,
    ) -> None:
        result = connection.execute(
            """
            update video_sequences
            set document_json = ?, row_revision = ?, status = ?,
                active_chain_revision = ?, updated_at = ?
            where sequence_id = ? and row_revision = ?
            """,
            (
                self._encode(value),
                value.revision,
                value.status,
                value.active_chain_revision,
                value.updated_at.isoformat(),
                value.sequence_id,
                previous.revision,
            ),
        )
        if result.rowcount != 1:
            raise VideoSequenceConflictError("video sequence revision changed during update")

    @staticmethod
    def _encode(value: VideoSequence) -> str:
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode(value: str) -> VideoSequence:
        return VideoSequence.model_validate_json(value)
