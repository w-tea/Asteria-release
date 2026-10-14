"""Frozen image groups and append-only human corrections, separate from generation."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from app.schemas.image_conditions import ConditionAssessment, ConditionCorrection
from app.schemas.series_consistency import SeriesRequest
from app.services.image_conditions import condition_verdict


def now():
    return datetime.now(timezone.utc).isoformat()


class SeriesRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS image_series_checks (id TEXT PRIMARY KEY, document TEXT NOT NULL)')

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def create(self, request: SeriesRequest, hashes: list[str], provider: dict | None = None):
        if len(hashes) != len(request.candidates) + 1:
            raise ValueError('series_hash_count')
        document = {**request.model_dump(exclude={'candidates'}), 'id': uuid4().hex, 'revision': 1,
                    'created_at': now(), 'provider': provider or {},
                    'reference': {**request.reference.model_dump(), 'source_sha256': hashes[0]},
                    'members': [{**item.model_dump(), 'source_sha256': digest, 'status': 'pending',
                                 'assessment': None, 'receipt': None, 'corrections': [], 'verdict': None, 'error': None}
                                for item, digest in zip(request.candidates, hashes[1:], strict=True)]}
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO image_series_checks VALUES (?, ?)', (document['id'], json.dumps(document, ensure_ascii=False)))
        return document

    def list(self):
        with closing(self._connect()) as db:
            return [json.loads(row[0]) for row in db.execute('SELECT document FROM image_series_checks ORDER BY rowid DESC LIMIT 50')]

    def get(self, run_id: str):
        with closing(self._connect()) as db:
            return self._load(db, run_id)

    @staticmethod
    def _load(db, run_id):
        row = db.execute('SELECT document FROM image_series_checks WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise KeyError('series_not_found')
        return json.loads(row[0])

    def _change(self, run_id, index, change):
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            document = self._load(db, run_id)
            if not 0 <= index < len(document['members']):
                raise ValueError('series_member_unavailable')
            change(document, document['members'][index])
            document['revision'] += 1
            db.execute('UPDATE image_series_checks SET document=? WHERE id=?', (json.dumps(document, ensure_ascii=False), run_id))
        return document

    def claim(self, run_id, index):
        def change(document, member):
            if document.get('cancelled_at'):
                raise ValueError('series_cancelled')
            if member['status'] != 'pending':
                raise ValueError('series_already_attempted')
            member.update(status='running', started_at=now())
        return self._change(run_id, index, change)

    def cancel(self, run_id):
        def change(document, member):
            document['cancelled_at'] = document.get('cancelled_at') or now()
        return self._change(run_id, 0, change)

    def finish(self, run_id, index, assessment: ConditionAssessment | None = None, receipt: dict | None = None, *, error=None):
        def change(document, member):
            if member['status'] != 'running':
                raise ValueError('series_not_running')
            if assessment:
                ids = [item.id for item in assessment.items]
                if len(set(ids)) != len(ids) or set(ids) != {x['id'] for x in document['criteria']}:
                    raise ValueError('condition_response_ids')
                member.update(status='completed', assessment=assessment.model_dump(), receipt=receipt)
                member['verdict'] = condition_verdict(member)
            else:
                member.update(status='failed', error=error or 'check_failed')
            member['finished_at'] = now()
        return self._change(run_id, index, change)

    def correct(self, run_id, index, correction: ConditionCorrection):
        def change(document, member):
            if document['revision'] != correction.revision:
                raise ValueError('condition_changed')
            if member['status'] != 'completed' or correction.criterion_id not in {x['id'] for x in document['criteria']}:
                raise ValueError('condition_unknown_criterion')
            member['corrections'].append({**correction.model_dump(exclude={'revision'}), 'created_at': now()})
            member['verdict'] = condition_verdict(member)
        return self._change(run_id, index, change)
