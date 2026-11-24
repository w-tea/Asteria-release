"""Find saved inputs that still need their original media or registration."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator


_DRAFT_ACTIONS = (
    'generation.drafts.create', 'generation.drafts.update', 'generation.drafts.derive',
    'generation.history.restore_draft', 'media.drafts.save', 'media.history.restore_draft',
)
_FRAME_KEYS = {'first_frame_artifact_id', 'last_frame_artifact_id', 'source_artifact_id'}


def adopted_request_documents(connection: sqlite3.Connection) -> Iterator[dict]:
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    if 'assistant_drafts' in tables:
        for (raw,) in connection.execute('select document_json from assistant_drafts'):
            yield json.loads(raw)
    if 'action_receipts' in tables and 'assistant_drafts' in tables:
        parameters = ', '.join('?' for _ in _DRAFT_ACTIONS)
        active = {row[0] for row in connection.execute('select id from assistant_drafts')}
        query = f'select result_json from action_receipts where action in ({parameters})'
        for (raw,) in connection.execute(query, _DRAFT_ACTIONS):
            item = json.loads(raw)
            if item['id'] in active:
                yield item['document']
    if 'operations' in tables:
        for (raw,) in connection.execute("""
            select request_snapshot_json from operations
            where status in ('queued', 'preparing', 'running', 'cancelling')
        """):
            yield json.loads(raw)


def _referenced_artifacts(value) -> Iterator[str]:
    if isinstance(value, dict):
        for key in _FRAME_KEYS:
            reference = value.get(key)
            if isinstance(reference, str):
                yield reference
        clip_ids = value.get('source_artifact_ids')
        if isinstance(clip_ids, list):
            yield from (item for item in clip_ids if isinstance(item, str))
        references = value.get('references')
        if isinstance(references, list):
            for item in references:
                if isinstance(item, dict) and isinstance(item.get('artifact_id'), str):
                    yield item['artifact_id']
        for child in value.values():
            yield from _referenced_artifacts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _referenced_artifacts(child)


def adopted_artifact_owners(items: list[dict], documents: Iterable[dict]) -> list[int]:
    candidates = {
        artifact['artifact_id']: item['id']
        for item in items for artifact in item['artifacts']
        if artifact['kind'] in {'image', 'video', 'audio'}
    }
    if not candidates:
        return []
    return sorted({candidates[artifact_id] for doc in documents for artifact_id in _referenced_artifacts(doc)
                   if artifact_id in candidates})
