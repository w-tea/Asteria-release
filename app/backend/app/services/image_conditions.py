"""Image-specific semantic checks; separate from anatomy and generation metadata."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from app.adapters.image_check_image import prepare_image_check_image
from app.schemas.image_conditions import ConditionAssessment, ConditionCorrection, ConditionRequest


def registered_output_path(item: dict | None, output_path: str) -> str:
    if not item or item.get('media_kind', 'image') != 'image':
        raise ValueError('condition_image_unavailable')
    path = Path(output_path).resolve()
    for value in item.get('output_paths', []):
        if path == Path(value).resolve():
            return value
    raise ValueError('condition_image_unavailable')


def prepare_registered_image(item: dict | None, output_path: str, current_root: Path):
    path = registered_output_path(item, output_path)
    settings = item.get('settings', {})
    root = Path(settings.get('output_root') or settings.get('output_directory') or current_root)
    return prepare_image_check_image(path, output_root=root)


def validate_assessment(request: ConditionRequest, assessment: ConditionAssessment) -> ConditionAssessment:
    ids = [finding.id for finding in assessment.items]
    if len(set(ids)) != len(ids) or set(ids) != {item.id for item in request.criteria}:
        raise ValueError('condition_response_ids')
    return assessment


def condition_verdict(document: dict) -> str:
    overrides = {change['criterion_id']: change['verdict'] for change in document['corrections']}
    values = [overrides.get(item['id']) or item['verdict'] for item in document['assessment']['items']]
    return 'mismatch' if 'mismatch' in values else 'unknown' if 'unknown' in values else 'match'


class ImageConditionRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS image_condition_checks (
                id TEXT PRIMARY KEY, generation_id INTEGER NOT NULL, output_path TEXT NOT NULL,
                verdict TEXT NOT NULL, document TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS image_conditions_generation ON image_condition_checks(generation_id, output_path)')

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def create(self, request: ConditionRequest, assessment: ConditionAssessment, receipt: dict) -> dict:
        validate_assessment(request, assessment)
        document = {**request.model_dump(), 'id': uuid4().hex, 'revision': 1, 'assessment': assessment.model_dump(),
                    'receipt': receipt, 'corrections': [], 'created_at': datetime.now(timezone.utc).isoformat()}
        document['verdict'] = condition_verdict(document)
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO image_condition_checks VALUES (?, ?, ?, ?, ?)',
                       (document['id'], request.generation_id, request.output_path, document['verdict'], json.dumps(document, ensure_ascii=False)))
        return document

    def list(self, generation_id: int, output_path: str) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute('SELECT document FROM image_condition_checks WHERE generation_id=? AND output_path=? ORDER BY rowid DESC LIMIT 100',
                              (generation_id, output_path)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get(self, check_id: str) -> dict:
        with closing(self._connect()) as db:
            row = db.execute('SELECT document FROM image_condition_checks WHERE id=?', (check_id,)).fetchone()
        if row is None:
            raise KeyError('condition_not_found')
        return json.loads(row[0])

    def correct(self, check_id: str, correction: ConditionCorrection) -> dict:
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT document FROM image_condition_checks WHERE id=?', (check_id,)).fetchone()
            if row is None:
                raise KeyError('condition_not_found')
            document = json.loads(row[0])
            if document['revision'] != correction.revision:
                raise ValueError('condition_changed')
            if correction.criterion_id not in {item['id'] for item in document['criteria']}:
                raise ValueError('condition_unknown_criterion')
            document['corrections'].append({**correction.model_dump(exclude={'revision'}), 'created_at': datetime.now(timezone.utc).isoformat()})
            document['revision'] += 1
            document['verdict'] = condition_verdict(document)
            db.execute('UPDATE image_condition_checks SET verdict=?, document=? WHERE id=?',
                       (document['verdict'], json.dumps(document, ensure_ascii=False), check_id))
        return document


def condition_filter_clause(db_path: Path, verdict: str) -> tuple[str, list]:
    if verdict not in {'match', 'mismatch', 'unknown', 'unchecked'}:
        raise ValueError('invalid_condition_filter')
    ImageConditionRepository(db_path)
    # One latest check per registered output. A multi-output generation may match
    # several filters; the selected output's own report remains the authority.
    expression = '''SELECT 1 FROM image_condition_checks c WHERE c.generation_id=g.id
        AND c.output_path IN (SELECT value FROM json_each(g.output_paths_json))
        AND NOT EXISTS (SELECT 1 FROM image_condition_checks n WHERE n.generation_id=c.generation_id
            AND n.output_path=c.output_path AND n.rowid>c.rowid)'''
    if verdict == 'unchecked':
        return f"g.media_kind='image' AND NOT EXISTS ({expression})", []
    return f'EXISTS ({expression} AND c.verdict=?)', [verdict]
