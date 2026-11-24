"""Publish request-owned copies through the ordinary Library ZIP writer."""
from __future__ import annotations

from contextlib import closing
import logging
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from app.schemas.action_contracts import ActionResult, GenerationHistoryGet, LibraryExportResult
from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
from app.services.history import HistoryRepository
from app.services.operation_repository import canonical_operation_json
from app.services.reference_mutation import reference_mutation_coordinator
from app.services.resource_lifecycle import ResourceAdmissionTimeoutError, resource_admission


_ACTION = 'library.items.export'
logger = logging.getLogger(__name__)


def _bundle_path(data_dir: Path, generation_id: int, export_id: str) -> Path:
    return data_dir / 'exports' / export_id / f'generation-{generation_id}.zip'


def export_library_item(data_dir: Path, request: GenerationHistoryGet, request_id: str) -> ActionResult:
    from app.api.generation import _write_generation_bundle
    from app.services.action_dispatch import ActionError

    arguments = canonical_operation_json(request.model_dump(mode='json'))
    try:
        lease = resource_admission.acquire((), operation_id=_ACTION, timeout_seconds=0)
    except ResourceAdmissionTimeoutError as exc:
        raise ActionError('maintenance_restore_busy', 409) from exc
    with lease, reference_mutation_coordinator.acquire_publisher():
        history = HistoryRepository(data_dir / 'asteria.db')
        with history.publication() as connection:
            create_action_receipt_table(connection)
            previous = read_action_receipt(connection, request_id, _ACTION, arguments)
            if previous is not None:
                return ActionResult(data=previous, replayed=True)
            item = history.get(request.generation_id)
            if item is None:
                raise ActionError('generation_not_found', 404)
            export_id = uuid4().hex
            pending = LibraryExportResult(generation_id=request.generation_id, export_id=export_id, status='unconfirmed')
            save_action_receipt(connection, request_id, _ACTION, arguments, pending.model_dump_json())

        path = _bundle_path(data_dir, request.generation_id, export_id)
        bundle = _write_generation_bundle(history, item, path, unique_directory=True)
        result = LibraryExportResult(
            generation_id=request.generation_id, export_id=export_id, status='completed',
            download_path=f'/api/generations/history/{request.generation_id}/exports/{export_id}',
            file_count=bundle.file_count, missing_count=len(bundle.missing_files),
            bundle_size_bytes=bundle.bundle_size_bytes, bundle_sha256=bundle.bundle_sha256,
        )
        try:
            with history.publication() as connection:
                connection.execute('update action_receipts set result_json=? where request_id=? and action=?',
                                   (result.model_dump_json(), request_id, _ACTION))
        except (OSError, sqlite3.Error):
            # A published archive stays private to its known reservation, and
            # a transport retry never runs the export again.
            logger.warning('Could not record the completed Library export', exc_info=True)
            return ActionResult(data=pending.model_dump(mode='json'))
        return ActionResult(data=result.model_dump(mode='json'))


def resolve_library_export(data_dir: Path, generation_id: int, export_id: str) -> Path | None:
    if generation_id < 1 or not re.fullmatch(r'[a-f0-9]{32}', export_id):
        return None
    db = data_dir / 'asteria.db'
    if not db.is_file():
        return None
    with closing(sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        if not connection.execute("select 1 from sqlite_master where name = 'action_receipts' and type = 'table'").fetchone():
            return None
        receipt = connection.execute(
            """select result_json from action_receipts where action=?
                and json_extract(result_json, '$.export_id')=?
                and json_extract(result_json, '$.generation_id')=?""",
            (_ACTION, export_id, generation_id),
        ).fetchone()
    if receipt is None:
        return None
    export = LibraryExportResult.model_validate_json(receipt[0])
    if export.status != 'completed':
        return None
    path = _bundle_path(data_dir, generation_id, export_id)
    try:
        root = (data_dir / 'exports').resolve(strict=True)
        if (path.resolve(strict=True) != root / export_id / path.name
                or not path.is_file() or path.stat().st_size != export.bundle_size_bytes):
            return None
    except (OSError, RuntimeError):
        return None
    return path
