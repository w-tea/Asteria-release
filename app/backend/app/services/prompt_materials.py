"""Small reusable prompt materials, separate from full generation presets."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from app.schemas.prompt_materials import PromptMaterialWrite


class PromptMaterialRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS prompt_materials (id TEXT PRIMARY KEY, document TEXT NOT NULL)')

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def list(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute('SELECT document FROM prompt_materials ORDER BY rowid DESC').fetchall()
        return [json.loads(row[0]) for row in rows]

    def create(self, material: PromptMaterialWrite) -> dict:
        document = {**material.model_dump(), 'id': uuid4().hex, 'revision': 1,
                    'updated_at': datetime.now(timezone.utc).isoformat()}
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO prompt_materials VALUES (?, ?)', (document['id'], json.dumps(document, ensure_ascii=False)))
        return document

    def _existing(self, db, material_id: str, revision: int) -> dict:
        row = db.execute('SELECT document FROM prompt_materials WHERE id=?', (material_id,)).fetchone()
        if row is None:
            raise KeyError('material_not_found')
        document = json.loads(row[0])
        if document['revision'] != revision:
            raise ValueError('material_changed')
        return document

    def update(self, material_id: str, material: PromptMaterialWrite, revision: int) -> dict:
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            old = self._existing(db, material_id, revision)
            document = {**old, **material.model_dump(), 'revision': revision + 1,
                        'updated_at': datetime.now(timezone.utc).isoformat()}
            db.execute('UPDATE prompt_materials SET document=? WHERE id=?', (json.dumps(document, ensure_ascii=False), material_id))
        return document

    def delete(self, material_id: str, revision: int) -> None:
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            self._existing(db, material_id, revision)
            db.execute('DELETE FROM prompt_materials WHERE id=?', (material_id,))
