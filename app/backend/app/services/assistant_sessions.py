"""Persist chats and streamed events without creating a second generation queue."""
from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from app.schemas.assistant import AssistantEvent, AssistantTurn, Conversation, TurnStatus
from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
from app.services.operation_repository import canonical_operation_json
from app.services.assistant_drafts import AssistantDraftRepository, DraftConflict, DraftNotFound


class ConversationNotFound(LookupError):
    pass


class ConversationAnswerActive(ValueError):
    pass


class ConversationConflict(ValueError):
    def __init__(self, current: Conversation):
        super().__init__('assistant_conversation_revision_conflict')
        self.current = current


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AssistantSessionRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript('''
                create table if not exists assistant_conversations (
                    id text primary key, revision integer not null,
                    title text not null, provider_id text not null, model text not null,
                    created_at text not null, updated_at text not null
                );
                create table if not exists assistant_turns (
                    id text primary key, conversation_id text not null,
                    prompt text not null, provider_id text not null, model text not null,
                    status text not null, created_at text not null, updated_at text not null
                );
                create index if not exists assistant_turn_conversation on assistant_turns(conversation_id, created_at);
                create table if not exists assistant_events (
                    turn_id text not null, sequence integer not null,
                    kind text not null, payload_json text not null, created_at text not null,
                    primary key(turn_id, sequence)
                );
            ''')
            create_action_receipt_table(con)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _get(con: sqlite3.Connection, table: str, identifier: str, model):
        # Table names are internal constants, never request input.
        row = con.execute(f'select * from {table} where id=?', (identifier,)).fetchone()
        if row is None:
            raise ConversationNotFound(identifier)
        return model.model_validate(dict(row))

    def get(self, conversation_id: str) -> Conversation:
        with closing(self._connect()) as con:
            return self._get(con, 'assistant_conversations', conversation_id, Conversation)

    def get_turn(self, turn_id: str) -> AssistantTurn:
        with closing(self._connect()) as con:
            return self._get(con, 'assistant_turns', turn_id, AssistantTurn)

    def list(self, query: str = '') -> list[Conversation]:
        with closing(self._connect()) as con:
            return [Conversation.model_validate(dict(row)) for row in con.execute(
                'select * from assistant_conversations where instr(lower(title), lower(?)) > 0 order by updated_at desc, id',
                (query,),
            )]

    def create(self, request_id: str, *, title: str, provider_id: str, model: str) -> tuple[Conversation, bool]:
        if not request_id:
            raise ValueError('action_request_id_required')
        args = canonical_operation_json(dict(title=title, provider_id=provider_id, model=model))
        with closing(self._connect()) as con, con:
            con.execute('begin immediate')
            receipt = read_action_receipt(con, request_id, 'assistant.conversations.create', args)
            if receipt:
                return self._get(con, 'assistant_conversations', receipt['id'], Conversation), True
            identifier, now = str(uuid4()), _now()
            con.execute('insert into assistant_conversations values (?, 1, ?, ?, ?, ?, ?)',
                        (identifier, title, provider_id, model, now, now))
            save_action_receipt(con, request_id, 'assistant.conversations.create', args, json.dumps({'id': identifier}))
            return self._get(con, 'assistant_conversations', identifier, Conversation), False

    def update(self, conversation_id: str, revision: int, *, title: str, provider_id: str, model: str) -> Conversation:
        with closing(self._connect()) as con, con:
            con.execute('begin immediate')
            current = self._get(con, 'assistant_conversations', conversation_id, Conversation)
            if current.revision != revision:
                raise ConversationConflict(current)
            con.execute('update assistant_conversations set revision=revision+1, title=?, provider_id=?, model=?, updated_at=? where id=?',
                        (title, provider_id, model, _now(), conversation_id))
            return self._get(con, 'assistant_conversations', conversation_id, Conversation)

    def delete(self, conversation_id: str, revision: int, *, stream_pending) -> None:
        with closing(self._connect()) as con, con:
            con.execute('begin immediate')
            current = self._get(con, 'assistant_conversations', conversation_id, Conversation)
            if current.revision != revision:
                raise ConversationConflict(current)
            turns = con.execute('select id, status from assistant_turns where conversation_id=?', (conversation_id,)).fetchall()
            if any(row['status'] == 'running' or stream_pending(row['id']) for row in turns):
                raise ConversationAnswerActive()
            con.execute('delete from assistant_events where turn_id in (select id from assistant_turns where conversation_id=?)',
                        (conversation_id,))
            con.execute('delete from assistant_turns where conversation_id=?', (conversation_id,))
            con.execute('delete from assistant_conversations where id=?', (conversation_id,))
            # Keep acceptance receipts so a delayed request cannot recreate the
            # chat. A turn, draft, output or upload receipt is not a draft/file deletion.

    def start_turn(self, conversation_id: str, request_id: str, prompt: str, provider_id: str, model: str, *,
                   attachments: list[str] | None = None, media_attachments: list[dict] | None = None,
                   before_create=None, before_create_media=None) -> tuple[AssistantTurn, bool]:
        if not request_id:
            raise ValueError('action_request_id_required')
        arguments = dict(conversation_id=conversation_id, prompt=prompt, provider_id=provider_id, model=model)
        if attachments:
            arguments['attachments'] = attachments
        if media_attachments:
            arguments['media_attachments'] = media_attachments
        args = canonical_operation_json(arguments)
        with closing(self._connect()) as con, con:
            con.execute('begin immediate')
            self._get(con, 'assistant_conversations', conversation_id, Conversation)
            receipt = read_action_receipt(con, request_id, 'assistant.turns.create', args)
            if receipt:
                return self._get(con, 'assistant_turns', receipt['id'], AssistantTurn), True
            images = before_create() if before_create is not None else None
            if attachments and (not images or [image['asset_id'] for image in images] != attachments):
                raise ValueError('assistant_attachments_not_resolved')
            media = before_create_media() if before_create_media is not None else []
            if media_attachments and [{key: item[key] for key in ('generation_id', 'artifact_id')} for item in media] != media_attachments:
                raise ValueError('assistant_attachments_not_resolved')
            identifier, now = str(uuid4()), _now()
            con.execute('insert into assistant_turns values (?, ?, ?, ?, ?, ?, ?, ?)',
                        (identifier, conversation_id, prompt, provider_id, model, 'running', now, now))
            con.execute('update assistant_conversations set updated_at=? where id=?', (now, conversation_id))
            if images:
                con.execute('insert into assistant_events values (?, 1, ?, ?, ?)',
                            (identifier, 'images_attached', canonical_operation_json({'images': images}), now))
            if media:
                con.execute('insert into assistant_events values (?, ?, ?, ?, ?)',
                            (identifier, 2 if images else 1, 'media_attached', canonical_operation_json({'media': media}), now))
            save_action_receipt(con, request_id, 'assistant.turns.create', args, json.dumps({'id': identifier}))
            return self._get(con, 'assistant_turns', identifier, AssistantTurn), False

    def turns(self, conversation_id: str) -> list[AssistantTurn]:
        with closing(self._connect()) as con:
            self._get(con, 'assistant_conversations', conversation_id, Conversation)
            return [AssistantTurn.model_validate(dict(row)) for row in con.execute(
                'select * from assistant_turns where conversation_id=? order by created_at, id', (conversation_id,),
            )]

    def append_event(self, turn_id: str, kind: str, payload: dict[str, Any]) -> AssistantEvent:
        with closing(self._connect()) as con, con:
            con.execute('begin immediate')
            self._get(con, 'assistant_turns', turn_id, AssistantTurn)
            sequence = con.execute('select coalesce(max(sequence), 0)+1 from assistant_events where turn_id=?', (turn_id,)).fetchone()[0]
            now = _now()
            con.execute('insert into assistant_events values (?, ?, ?, ?, ?)',
                        (turn_id, sequence, kind, canonical_operation_json(payload), now))
            return AssistantEvent(sequence=sequence, kind=kind, payload=payload, created_at=now)

    def events(self, turn_id: str, *, after: int = 0) -> list[AssistantEvent]:
        with closing(self._connect()) as con:
            self._get(con, 'assistant_turns', turn_id, AssistantTurn)
            return [AssistantEvent(sequence=row['sequence'], kind=row['kind'], payload=json.loads(row['payload_json']), created_at=row['created_at'])
                    for row in con.execute('select * from assistant_events where turn_id=? and sequence>? order by sequence', (turn_id, after))]

    def attach_draft(self, turn_id: str, request_id: str, draft_id: str, expected_revision: int) -> tuple[AssistantEvent, bool]:
        """Link an actual saved version, never client-supplied mask/settings data."""
        args = canonical_operation_json(dict(turn_id=turn_id, draft_id=draft_id, expected_revision=expected_revision))
        with closing(self._connect()) as con, con:
            con.execute('begin immediate')
            turn = self._get(con, 'assistant_turns', turn_id, AssistantTurn)
            receipt = read_action_receipt(con, request_id, 'assistant.drafts.attach', args)
            if receipt:
                return AssistantEvent.model_validate(receipt), True
            row = con.execute('select * from assistant_drafts where id=?', (draft_id,)).fetchone()
            if row is None:
                raise DraftNotFound(draft_id)
            draft = AssistantDraftRepository._record(row)
            if draft.revision != expected_revision:
                raise DraftConflict('assistant_draft_revision_conflict', draft)
            sequence = con.execute('select coalesce(max(sequence), 0)+1 from assistant_events where turn_id=?', (turn_id,)).fetchone()[0]
            now = _now()
            event = AssistantEvent(sequence=sequence, kind='draft_saved', payload={'draft': draft.model_dump()}, created_at=now)
            con.execute('insert into assistant_events values (?, ?, ?, ?, ?)',
                        (turn_id, sequence, event.kind, canonical_operation_json(event.payload), now))
            con.execute('update assistant_conversations set updated_at=? where id=?', (now, turn.conversation_id))
            save_action_receipt(con, request_id, 'assistant.drafts.attach', args, event.model_dump_json())
            return event, False

    def finish(self, turn_id: str, status: TurnStatus) -> AssistantTurn:
        if status not in {'completed', 'cancelled', 'failed', 'interrupted'}:
            raise ValueError('assistant_terminal_status_required')
        with closing(self._connect()) as con, con:
            con.execute("update assistant_turns set status=?, updated_at=? where id=? and status='running'", (status, _now(), turn_id))
            return self._get(con, 'assistant_turns', turn_id, AssistantTurn)

    def recover_interrupted(self) -> int:
        # A conversation is never replayed automatically after a server restart.
        with closing(self._connect()) as con, con:
            return con.execute("update assistant_turns set status='interrupted', updated_at=? where status='running'", (_now(),)).rowcount

    def export(self, conversation_id: str) -> dict[str, Any]:
        with closing(self._connect()) as con, con:
            con.execute('begin')
            conversation = self._get(con, 'assistant_conversations', conversation_id, Conversation)
            turns = con.execute('select * from assistant_turns where conversation_id=? order by created_at, id',
                                (conversation_id,)).fetchall()
            events: dict[str, list] = {turn['id']: [] for turn in turns}
            for row in con.execute('''select e.* from assistant_events e
                    join assistant_turns t on e.turn_id=t.id where t.conversation_id=?
                    order by e.turn_id, e.sequence''', (conversation_id,)):
                event = AssistantEvent(sequence=row['sequence'], kind=row['kind'], payload=json.loads(row['payload_json']),
                                       created_at=row['created_at'])
                events[row['turn_id']].append(event.model_dump())
            return {'conversation': conversation.model_dump(), 'turns': [
                {**AssistantTurn.model_validate(dict(turn)).model_dump(), 'events': events[turn['id']]}
                for turn in turns
            ]}
