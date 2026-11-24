"""Named generation drafts; never the browser's current Image Edit draft.

The draft and its retry receipt commit together in the existing application DB.
Request IDs identify redelivery, not clients, conversations or prompt contents.
"""
from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable
from uuid import uuid4

from pydantic import TypeAdapter

from app.schemas.action_contracts import AssistantDraftRecord, SavedGenerationDraftDocument
from app.services.operation_repository import canonical_operation_json
from app.services.reference_mutation import reference_mutation_coordinator
from app.services.action_receipts import (
    ActionRequestConflict, create_action_receipt_table, read_action_receipt, save_action_receipt,
)


class DraftNotFound(LookupError):
    pass


class DraftRevisionNotFound(LookupError):
    def __init__(self, draft_id: str, revision: int):
        super().__init__(draft_id)
        self.draft_id, self.revision = draft_id, revision


class DraftConflict(ValueError):
    def __init__(self, code: str, current: AssistantDraftRecord | None = None):
        super().__init__(code)
        self.code, self.current = code, current


_RESTORE_HISTORY_ACTION = 'generation.history.restore_draft'
_RESTORE_MEDIA_HISTORY_ACTION = 'media.history.restore_draft'


class AssistantDraftRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.execute('''create table if not exists assistant_drafts (
                id text primary key, revision integer not null,
                title text not null, document_json text not null,
                created_at text not null, updated_at text not null
            )''')
            create_action_receipt_table(con)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _record(row: sqlite3.Row) -> AssistantDraftRecord:
        return AssistantDraftRecord(
            id=row['id'], revision=row['revision'], title=row['title'],
            document=json.loads(row['document_json']),
            created_at=row['created_at'], updated_at=row['updated_at'],
        )

    def _get(self, con: sqlite3.Connection, draft_id: str) -> AssistantDraftRecord:
        row = con.execute('select * from assistant_drafts where id=?', (draft_id,)).fetchone()
        if row is None:
            raise DraftNotFound(draft_id)
        return self._record(row)

    def get(self, draft_id: str) -> AssistantDraftRecord:
        with closing(self._connect()) as con:
            return self._get(con, draft_id)

    def get_revision(self, draft_id: str, revision: int) -> AssistantDraftRecord:
        with closing(self._connect()) as con:
            current = self._get(con, draft_id)
            if current.revision == revision:
                return current
            # Each accepted save already keeps its complete result in the same
            # transaction. Read that immutable version, not another client's latest.
            row = con.execute('''select result_json from action_receipts
                where action in ('generation.drafts.create', 'generation.drafts.update', 'generation.drafts.derive',
                                 'media.drafts.save', ?, ?)
                and json_extract(case when json_valid(result_json) then result_json else '{}' end, '$.id') = ?
                and json_extract(case when json_valid(result_json) then result_json else '{}' end, '$.revision') = ?
                limit 1''', (_RESTORE_HISTORY_ACTION, _RESTORE_MEDIA_HISTORY_ACTION, draft_id, revision)).fetchone()
            if row is None:
                raise DraftRevisionNotFound(draft_id, revision)
            return AssistantDraftRecord.model_validate_json(row['result_json'])

    def list(self) -> list[AssistantDraftRecord]:
        with closing(self._connect()) as con:
            return [self._record(row) for row in con.execute(
                'select * from assistant_drafts order by updated_at desc, id'
            )]

    def _write(self, request_id: str, action: str, arguments: dict[str, Any],
               write: Callable[[sqlite3.Connection], AssistantDraftRecord]) -> tuple[AssistantDraftRecord, bool]:
        if not request_id:
            raise ValueError('action_request_id_required')
        canonical = canonical_operation_json(arguments)
        with reference_mutation_coordinator.acquire_publisher(), closing(self._connect()) as con, con:
            con.execute('begin immediate')
            try:
                receipt = read_action_receipt(con, request_id, action, canonical)
            except ActionRequestConflict as exc:
                raise DraftConflict('action_request_conflict') from exc
            if receipt is not None:
                return AssistantDraftRecord.model_validate(receipt), True
            result = write(con)
            save_action_receipt(con, request_id, action, canonical, result.model_dump_json())
            return result, False

    def _insert(self, con: sqlite3.Connection, title: str, document: dict[str, Any]) -> AssistantDraftRecord:
        now, draft_id = datetime.now(UTC).isoformat(), str(uuid4())
        con.execute('insert into assistant_drafts values (?, 1, ?, ?, ?, ?)',
                    (draft_id, title, canonical_operation_json(document), now, now))
        return self._get(con, draft_id)

    def create(self, request_id: str, title: str, document: dict[str, Any], *,
               action: str = 'generation.drafts.create') -> tuple[AssistantDraftRecord, bool]:
        return self._write(request_id, action, {'title': title, 'document': document},
                           lambda con: self._insert(con, title, document))

    def restore_history(self, request_id: str, generation_id: int, title: str,
                        document: Callable[[], dict[str, Any]], *,
                        output_index: int | None = None) -> tuple[AssistantDraftRecord, bool]:
        arguments = dict(generation_id=generation_id, title=title)
        if output_index is not None:
            arguments['output_index'] = output_index
        return self._write(request_id, _RESTORE_HISTORY_ACTION, arguments,
                           lambda con: self._insert(con, title, document()))

    def restore_media_history(self, request_id: str, kind: str, generation_id: int, title: str,
                              document: Callable[[], dict[str, Any]]) -> tuple[AssistantDraftRecord, bool]:
        return self._write(request_id, _RESTORE_MEDIA_HISTORY_ACTION,
                           dict(kind=kind, generation_id=generation_id, title=title),
                           lambda con: self._insert(con, title, document()))

    def update(self, request_id: str, draft_id: str, expected_revision: int,
               title: str, document: dict[str, Any], *, action: str = 'generation.drafts.update') -> tuple[AssistantDraftRecord, bool]:
        def write(con: sqlite3.Connection) -> AssistantDraftRecord:
            current = self._get(con, draft_id)
            if current.revision != expected_revision:
                raise DraftConflict('assistant_draft_revision_conflict', current)
            if current.document.get('kind', 'image_generation') != document.get('kind', 'image_generation'):
                raise DraftConflict('assistant_draft_kind_mismatch', current)
            con.execute('update assistant_drafts set revision=revision+1, title=?, document_json=?, updated_at=? where id=?',
                        (title, canonical_operation_json(document), datetime.now(UTC).isoformat(), draft_id))
            return self._get(con, draft_id)
        return self._write(request_id, action, dict(draft_id=draft_id, expected_revision=expected_revision,
                                                    title=title, document=document), write)

    def derive(self, request_id: str, source_draft_id: str, expected_revision: int,
               title: str, input_changes: dict[str, Any]) -> tuple[AssistantDraftRecord, bool]:
        def write(con: sqlite3.Connection) -> AssistantDraftRecord:
            source = self._get(con, source_draft_id)
            if source.revision != expected_revision:
                raise DraftConflict('assistant_draft_revision_conflict', source)
            adapter = TypeAdapter(SavedGenerationDraftDocument)
            saved = adapter.validate_python(source.document)
            unknown = input_changes.keys() - type(saved.input).model_fields.keys()
            if unknown:
                raise DraftConflict('assistant_draft_input_field_unknown')
            document = {**source.document, 'input': {**source.document['input'], **input_changes}}
            validated = adapter.validate_python(document).model_dump(mode='json')
            return self._insert(con, title, validated)
        return self._write(request_id, 'generation.drafts.derive', dict(
            source_draft_id=source_draft_id, expected_revision=expected_revision,
            title=title, input_changes=input_changes), write)
