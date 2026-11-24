"""Saved image inputs shared by the core action API and MCP.

Bytes are validated and stored by the ordinary asset uploader. A successful
receipt and its asset identity commit together; no current editor draft changes.
"""
from __future__ import annotations

import base64
from contextlib import closing
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sqlite3
from uuid import uuid4

from fastapi import HTTPException
from PIL import Image

from app.schemas.action_contracts import ActionImageContent, ActionImageRecord, ImageUploadMetadata, InlineImage
from app.schemas.media import image_media_type
from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
from app.services.operation_repository import canonical_operation_json
from app.services.reference_mutation import reference_mutation_coordinator


class ActionImageRepository:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.db_path = data_dir / 'asteria.db'
        data_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.execute('create table if not exists assistant_image_assets (id text primary key, record_json text not null)')
            create_action_receipt_table(con)

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def save(self, request: ImageUploadMetadata, request_id: str, data: bytes) -> tuple[ActionImageRecord, bool]:
        from app.api.assets import save_uploaded_image_bytes

        fingerprint = sha256(data).hexdigest()
        arguments = canonical_operation_json({'filename': request.filename, 'role': request.role, 'sha256': fingerprint})
        saved = record = None
        try:
            with reference_mutation_coordinator.acquire_publisher(), closing(self._connect()) as con, con:
                con.execute('begin immediate')
                receipt = read_action_receipt(con, request_id, 'images.import', arguments)
                if receipt is not None:
                    return ActionImageRecord.model_validate(receipt), True
                saved = save_uploaded_image_bytes(data, request.filename, request.role + '_images', request.role,
                                                   data_dir=self.data_dir)
                with Image.open(BytesIO(data)) as image:
                    width, height = image.size
                record = ActionImageRecord(asset_id=uuid4().hex, path=saved.path, filename=saved.filename,
                    role=request.role, mime_type=image_media_type(saved.path), sha256=fingerprint, width=width, height=height)
                con.execute('insert into assistant_image_assets values (?, ?)', (record.asset_id, record.model_dump_json()))
                save_action_receipt(con, request_id, 'images.import', arguments, record.model_dump_json())
            return record, False
        except BaseException:
            # Only this call's new, randomly named file can be removed. A process
            # crash can leave an unreferenced upload for the existing age-based
            # asset cleanup, but never a receipt for an unfinished file.
            if saved is not None:
                # An interruption can arrive just after SQLite commits. Verify
                # absence before removing anything; uncertainty preserves it.
                try:
                    with closing(self._connect()) as con:
                        accepted = record is not None and con.execute(
                            'select 1 from assistant_image_assets where id=?', (record.asset_id,)).fetchone()
                except (OSError, sqlite3.Error):
                    accepted = True
                try:
                    if not accepted:
                        Path(saved.path).unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def read_bytes(self, asset_id: str) -> tuple[ActionImageRecord, bytes]:
        with closing(self._connect()) as con:
            row = con.execute('select record_json from assistant_image_assets where id=?', (asset_id,)).fetchone()
        if row is None:
            raise HTTPException(404, detail={'code': 'asset_not_found'})
        record = ActionImageRecord.model_validate_json(row[0])
        root = (self.data_dir / (record.role + '_images')).resolve()
        try:
            path = Path(record.path).resolve(strict=True)
            if path.parent != root or path.name != record.filename:
                raise OSError('asset path changed')
            data = path.read_bytes()
        except (OSError, RuntimeError):
            raise HTTPException(404, detail={'code': 'asset_not_found'}) from None
        if sha256(data).hexdigest() != record.sha256:
            raise HTTPException(409, detail={'code': 'assistant_image_changed', 'asset_id': asset_id})
        return record, data

    def get(self, asset_id: str) -> ActionImageContent:
        record, data = self.read_bytes(asset_id)
        return ActionImageContent(**record.model_dump(), image=InlineImage(
            mime_type=record.mime_type, data_base64=base64.b64encode(data).decode('ascii')))

    def get_by_saved_path(self, path: Path) -> ActionImageContent | None:
        """Keep upload identity/hash verification for saved drafts storing paths."""
        with closing(self._connect()) as con:
            rows = con.execute('''select id, record_json from assistant_image_assets
                where json_extract(record_json, '$.filename') = ? collate nocase''', (path.name,)).fetchall()
        for identity, raw in rows:
            record = ActionImageRecord.model_validate_json(raw)
            if Path(record.path).resolve() == path:
                return self.get(identity)
        return None
