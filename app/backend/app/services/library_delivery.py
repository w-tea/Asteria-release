"""Common Library entry points for the configured Eagle and Topaz integrations."""
from __future__ import annotations

import logging
from pathlib import Path
import sqlite3

from fastapi import HTTPException

from app.core.config import AppConfig
from app.schemas.action_contracts import ActionResult, DraftList, GenerationJobGet, MediaSourceTarget
from app.schemas.library_delivery import LibraryEagleResult, LibraryEagleSend, LibraryTopazJob, LibraryTopazStart
from app.schemas.operations import OperationRecord
from app.services.action_dispatch import ActionError, _Action
from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
from app.services.history import HistoryRepository
from app.services.library_image_source import resolve_library_integration_image
from app.services.media_artifacts import MediaArtifactAccessError
from app.services.operation_repository import OperationNotFoundError, canonical_operation_json
from app.services.reference_mutation import reference_mutation_coordinator
from app.services.resource_lifecycle import ResourceAdmissionTimeoutError, resource_admission


_EAGLE_SEND = 'library.eagle.send'
_TOPAZ_START = 'library.topaz.start'
logger = logging.getLogger(__name__)


class LibraryDelivery:
    def __init__(self, config: AppConfig):
        self.data_dir = config.data_dir

    def catalog(self):
        from app.api.eagle import EagleFolderListResponse

        return {
            'library.eagle.folders': _Action(
                'List configured Eagle folders',
                'Read folder names and IDs from the configured Eagle library to choose the import destination.',
                DraftList, EagleFolderListResponse, True, self.eagle_folders),
            _EAGLE_SEND: _Action(
                'Send a saved image to Eagle',
                'Send the exact selected Library image. Omit folder_id to use the configured destination. '
                'Use the same request_id when resending; if unconfirmed, check Eagle before making a new request.',
                LibraryEagleSend, LibraryEagleResult, False, self.eagle_send),
            _TOPAZ_START: _Action(
                'Process a saved image with Topaz',
                'Start the existing Topaz operation for the exact saved image, without replacing the source. '
                'Use library.topaz.get to check the job and find the registered result.',
                LibraryTopazStart, LibraryTopazJob, False, self.topaz_start),
            'library.topaz.get': _Action(
                'Read a Topaz operation and its saved result',
                'Check the selected Topaz job. A registered output can be opened using its history and artifact IDs.',
                GenerationJobGet, LibraryTopazJob, True, self.topaz_get),
        }

    def _check_runtime(self) -> None:
        if self.data_dir.resolve() != AppConfig().data_dir.resolve():
            raise ActionError('action_runtime_configuration_mismatch', 503)

    def _source(self, target: MediaSourceTarget) -> Path:
        try:
            return resolve_library_integration_image(self.data_dir, target)
        except MediaArtifactAccessError as exc:
            unavailable = {'generation_not_found', 'media_artifact_not_found', 'media_artifact_not_registered'}
            raise ActionError(exc.code, 404 if exc.code in unavailable else 409) from exc

    def eagle_folders(self, _request: DraftList, _request_id: str | None) -> ActionResult:
        from app.api import eagle

        self._check_runtime()
        try:
            return ActionResult(data=eagle.list_folders().model_dump(mode='json'))
        except HTTPException as exc:
            raise ActionError('eagle_folder_list_failed', exc.status_code) from exc

    def eagle_send(self, request: LibraryEagleSend, request_id: str) -> ActionResult:
        from app.api import eagle

        self._check_runtime()
        arguments = canonical_operation_json(request.model_dump(mode='json'))
        try:
            lease = resource_admission.acquire((), operation_id=_EAGLE_SEND, timeout_seconds=0)
        except ResourceAdmissionTimeoutError as exc:
            raise ActionError('maintenance_restore_busy', 409) from exc
        with lease, reference_mutation_coordinator.acquire_publisher():
            history = HistoryRepository(self.data_dir / 'asteria.db')
            with history.publication() as connection:
                create_action_receipt_table(connection)
                previous = read_action_receipt(connection, request_id, _EAGLE_SEND, arguments)
                if previous is not None:
                    return ActionResult(data=previous, replayed=True)
                path = self._source(request)
                pending = LibraryEagleResult(generation_id=request.generation_id, artifact_id=request.artifact_id,
                                             status='unconfirmed', folder_id=request.folder_id)
                save_action_receipt(connection, request_id, _EAGLE_SEND, arguments, pending.model_dump_json())

            try:
                response = eagle._import_resolved_output(
                    eagle.EagleImportRequest(filename=path.name, generation_id=request.generation_id,
                                             folder_id=request.folder_id, skip_duplicates=False),
                    path,
                )
                result = pending.model_copy(update={'status': 'success', 'folder_id': response.folder_id,
                                                     'folder_applied': response.folder_applied})
            except HTTPException as exc:
                # A lost response does not prove that Eagle rejected the image.
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                result = pending.model_copy(update={'error_code': detail.get('code', 'eagle_import_failed')})
            try:
                with history.publication() as connection:
                    connection.execute('update action_receipts set result_json=? where request_id=? and action=?',
                                       (result.model_dump_json(), request_id, _EAGLE_SEND))
            except (OSError, sqlite3.Error):
                logger.warning('Could not record the Eagle import result', exc_info=True)
                result = pending
            return ActionResult(data=result.model_dump(mode='json'))

    @staticmethod
    def _topaz_job(record: OperationRecord, *, replayed: bool = False) -> ActionResult:
        source = MediaSourceTarget.model_validate(record.request_snapshot['library_target'])
        artifact_id = next((ref for ref in record.artifact_refs if ref.startswith('media:')), None)
        output = MediaSourceTarget(generation_id=source.generation_id, artifact_id=artifact_id) if artifact_id else None
        job = LibraryTopazJob(job_id=record.operation_id, status=record.status, source=source, output=output,
                              error_code=record.error_code, warning_codes=record.warning_codes)
        return ActionResult(data=job.model_dump(mode='json'), replayed=replayed)

    def topaz_start(self, request: LibraryTopazStart, request_id: str) -> ActionResult:
        from app.api import topaz

        self._check_runtime()
        arguments = request.model_dump(mode='json')
        coordinator = topaz._topaz_operation_coordinator()
        previous = coordinator.repository.get_action_operation(request_id, _TOPAZ_START, arguments)
        if previous is not None:
            return self._topaz_job(previous, replayed=True)
        path = self._source(request)
        if not topaz.status().installed:
            raise ActionError('topaz_process_failed', 409)
        native_request = topaz.TopazProcessRequest(filename=path.name, generation_id=request.generation_id,
                                                   settings=request.settings)
        record, created = coordinator.enqueue_with_action_receipt(
            request_id=request_id, action=_TOPAZ_START, arguments=arguments, kind='topaz.process',
            source_actor='assistant', progress_total=1, progress_unit='image',
            request_snapshot={'request': native_request.model_dump(mode='json'),
                              'library_target': {'generation_id': request.generation_id,
                                                 'artifact_id': request.artifact_id}},
        )
        return self._topaz_job(record, replayed=not created)

    def topaz_get(self, request: GenerationJobGet, _request_id: str | None) -> ActionResult:
        from app.api import topaz

        self._check_runtime()
        try:
            record = topaz._topaz_operation_coordinator().get(request.job_id)
        except OperationNotFoundError as exc:
            raise ActionError('operation_not_found', 404) from exc
        if record.kind != 'topaz.process' or 'library_target' not in record.request_snapshot:
            raise ActionError('operation_not_found', 404)
        return self._topaz_job(record)
