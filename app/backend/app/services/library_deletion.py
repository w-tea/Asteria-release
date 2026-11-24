"""Shared action entry point for the existing Library history and file phases."""
from __future__ import annotations

import logging
from pathlib import Path
import sqlite3

from app.schemas.action_contracts import ActionResult, LibraryDelete, LibraryDeleteResult
from app.services import api_errors
from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
from app.services.history import HistoryAdoptedArtifactConflict, HistoryBulkDeleteConflictError, HistoryRepository
from app.services.operation_repository import canonical_operation_json
from app.services.resource_lifecycle import ResourceAdmissionTimeoutError, resource_admission


logger = logging.getLogger(__name__)


def delete_library_items(data_dir: Path, request: LibraryDelete, request_id: str) -> ActionResult:
    from app.api.generation import _delete_history_artifacts
    from app.services.action_dispatch import ActionError

    name = 'library.items.delete'
    arguments = canonical_operation_json(request.model_dump(mode='json'))
    try:
        lease = resource_admission.acquire((), operation_id=name, timeout_seconds=0)
    except ResourceAdmissionTimeoutError as exc:
        raise ActionError('maintenance_restore_busy', 409) from exc
    with lease:
        history = HistoryRepository(data_dir / 'asteria.db')
        with history.publication() as connection:
            create_action_receipt_table(connection)
            received = read_action_receipt(connection, request_id, name, arguments)
            if received is not None:
                return ActionResult(data=received, replayed=True)
            try:
                deleted = history.bulk_delete(request.generation_ids)
            except HistoryAdoptedArtifactConflict as exc:
                raise ActionError(str(exc), 409, generation_ids=exc.generation_ids) from exc
            except HistoryBulkDeleteConflictError as exc:
                raise ActionError(api_errors.HISTORY_BULK_DELETE_CONFLICT, 409, missing_ids=exc.missing_ids) from exc
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                raise ActionError(api_errors.HISTORY_BULK_DELETE_FAILED, 409) from exc
            result = LibraryDeleteResult(deleted_ids=[item['id'] for item in deleted],
                files_status='unconfirmed' if request.delete_files else 'not_requested')
            save_action_receipt(connection, request_id, name, arguments, result.model_dump_json())
        if request.delete_files:
            files = _delete_history_artifacts(deleted, data_dir, history)
            result = LibraryDeleteResult(deleted_ids=result.deleted_ids, files_status='finished',
                deleted_files=len(files.deleted_files), missing_files=len(files.missing_files),
                skipped_files=len(files.skipped_files), protected_files=len(files.protected_files))
            try:
                with history.publication() as connection:
                    # Finalize the receipt; a concurrent replay never enters the file phase.
                    connection.execute('update action_receipts set result_json=? where request_id=? and action=?',
                                       (result.model_dump_json(), request_id, name))
            except (OSError, sqlite3.Error):
                logger.warning('Could not record the file result after deleting Library history', exc_info=True)
        return ActionResult(data=result.model_dump(mode='json'))
