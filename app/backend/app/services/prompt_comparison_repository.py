"""Persist immutable comparison inputs and independently updated results/ratings."""
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3

from app.schemas.prompt_comparison import PromptComparisonRating
from app.schemas.lora_practical_test import PracticalTestRating
from app.services.action_receipts import (
    ActionRequestIdentity, create_action_receipt_table, read_action_receipt, save_action_receipt,
)
from app.services.operation_repository import canonical_operation_json


class PromptComparisonRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS prompt_comparisons (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, document TEXT NOT NULL)')

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def create(self, document: dict) -> dict:
        with closing(self._connect()) as db, db:
            self._insert(db, document)
        return document

    @staticmethod
    def _insert(db: sqlite3.Connection, document: dict) -> None:
        db.execute('INSERT INTO prompt_comparisons VALUES (?, ?, ?)',
                   (document['id'], document['created_at'], json.dumps(document, ensure_ascii=False)))

    def find_action_comparison(self, identity: ActionRequestIdentity) -> str | None:
        with closing(self._connect()) as db, db:
            create_action_receipt_table(db)
            receipt = read_action_receipt(db, identity.request_id, identity.action, canonical_operation_json(identity.arguments))
        return None if receipt is None else receipt['comparison_id']

    def create_for_action(self, document: dict, identity: ActionRequestIdentity) -> tuple[str, bool]:
        arguments = canonical_operation_json(identity.arguments)
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            create_action_receipt_table(db)
            receipt = read_action_receipt(db, identity.request_id, identity.action, arguments)
            if receipt is not None:
                return receipt['comparison_id'], True
            self._insert(db, document)
            save_action_receipt(db, identity.request_id, identity.action, arguments,
                                canonical_operation_json({'comparison_id': document['id']}))
        return document['id'], False

    def get(self, comparison_id: str) -> dict:
        with closing(self._connect()) as db:
            row = db.execute('SELECT document FROM prompt_comparisons WHERE id=?', (comparison_id,)).fetchone()
        if row is None:
            raise KeyError('comparison_not_found')
        return json.loads(row[0])

    def list_recent(self, limit: int = 30) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute('SELECT document FROM prompt_comparisons ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
        return [{key: value for key, value in json.loads(row[0]).items()
                 if key in {'id', 'name', 'created_at', 'status', 'seeds', 'change', 'operation_id', 'error'}} for row in rows]

    def list_practical(self, run_id: str) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT document FROM prompt_comparisons WHERE json_extract(document, '$.lora_test.run_id')=? ORDER BY created_at DESC LIMIT 30", (run_id,)).fetchall()
        return [{key: value for key, value in json.loads(row[0]).items()
                 if key in {'id', 'name', 'created_at', 'status'}} for row in rows]

    def _mutate(self, comparison_id: str, update: Callable[[dict], None]) -> dict:
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            return self._mutate_in_connection(db, comparison_id, update)

    @staticmethod
    def _mutate_in_connection(db: sqlite3.Connection, comparison_id: str, update: Callable[[dict], None]) -> dict:
        row = db.execute('SELECT document FROM prompt_comparisons WHERE id=?', (comparison_id,)).fetchone()
        if row is None:
            raise KeyError('comparison_not_found')
        document = json.loads(row[0])
        update(document)
        db.execute('UPDATE prompt_comparisons SET document=? WHERE id=?', (json.dumps(document, ensure_ascii=False), comparison_id))
        return document

    def claim_start(self, comparison_id: str, operation_id: str, *, connection: sqlite3.Connection | None = None) -> dict:
        def update(document):
            if document['status'] != 'prepared':
                raise ValueError('comparison_already_started')
            document.update(status='queued', operation_id=operation_id)
        if connection is not None:
            return self._mutate_in_connection(connection, comparison_id, update)
        return self._mutate(comparison_id, update)

    def set_status(self, comparison_id: str, status: str, error: str = '') -> dict:
        return self._mutate(comparison_id, lambda doc: doc.update(status=status, error=error))

    def update_cell(self, comparison_id: str, index: int, **result) -> dict:
        allowed = {'status', 'generation_id', 'output_paths', 'error'}
        if not result.keys() <= allowed:
            raise ValueError('comparison_input_is_frozen')
        return self._mutate(comparison_id, lambda doc: doc['cells'][index].update(result))

    def rate(self, comparison_id: str, rating: PromptComparisonRating) -> dict:
        def update(document):
            cells = [cell for cell in document['cells'] if cell['pair'] == rating.pair]
            if len(cells) != 2 or any(cell['status'] != 'completed' for cell in cells):
                raise ValueError('comparison_pair_incomplete')
            previous = document['ratings'].get(str(rating.pair), {})
            document['ratings'][str(rating.pair)] = {**rating.model_dump(), 'revealed': previous.get('revealed', False) or rating.revealed}
        return self._mutate(comparison_id, update)

    def rate_practical(self, comparison_id: str, rating: PracticalTestRating) -> dict:
        def update(document):
            if not document.get('lora_test'):
                raise ValueError('practical_not_found')
            cells = [cell for cell in document['cells'] if cell['pair'] == rating.pair]
            if len(cells) != 2 or any(cell['status'] != 'completed' for cell in cells):
                raise ValueError('comparison_pair_incomplete')
            review = document['lora_test']
            previous = review['ratings'].get(str(rating.pair), {})
            saved = {**rating.model_dump(), 'revealed': previous.get('revealed', False) or rating.revealed,
                     'saved_at': datetime.now(UTC).isoformat()}
            review['ratings'][str(rating.pair)] = saved
            review['rating_history'].append(saved)
        return self._mutate(comparison_id, update)
