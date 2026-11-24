"""Use saved Library images with the existing body-part review and mask service."""
from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel
from PIL import Image

from app.core.config import AppConfig
from app.schemas.action_contracts import ActionResult, ImageUploadMetadata, InlineImage
from app.schemas.body_review import CandidateBox, Part
from app.schemas.library_body_review import (
    LibraryPartCrop, LibraryPartCropResult, LibraryPartReviewItem, LibraryPartSnapshot, LibraryPartSource,
    LibraryPartDetect, LibraryPartDetectionResult, LibraryPartMaskPrepare, LibraryPartMaskResult,
    LibraryPartsCompare, LibraryPartsResult,
)
from app.services.action_dispatch import ActionError
from app.services.action_images import ActionImageRepository
from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
from app.services.body_review import BodyReviewService, mask_union_png
from app.services.history import HistoryRepository
from app.services.media_artifacts import MediaArtifactAccessError, resolve_registered_media_artifact
from app.services.operation_repository import canonical_operation_json
from app.services.reference_mutation import reference_mutation_coordinator
from app.services.resource_lifecycle import ResourceAdmissionTimeoutError, resource_admission
from app.services.user_settings import resolve_current_output_root


class LibraryBodyReviews:
    def __init__(self, config: AppConfig):
        self.db = config.data_dir / 'asteria.db'
        self.data_dir = config.data_dir
        self.review = BodyReviewService(config.data_dir, config.model_dir)

    def _source(self, item: dict | None, selection: LibraryPartSource) -> Path:
        from app.api.generation import _known_output_roots

        if item is None:
            raise ActionError('generation_not_found', 404, generation_id=selection.generation_id)
        if item['media_kind'] != 'image':
            raise ActionError('body_review_source_unavailable', 404, generation_id=selection.generation_id)
        try:
            _, source = resolve_registered_media_artifact(
                item, selection.artifact_id, accepted_roots=_known_output_roots(self.data_dir, [item]),
                allowed_kinds={'image'},
            )
        except MediaArtifactAccessError as exc:
            raise ActionError('body_review_source_unavailable', 404, generation_id=selection.generation_id) from exc
        return source

    def _one_source(self, selection: LibraryPartSource) -> Path:
        history = HistoryRepository.get_many_existing(self.db, [selection.generation_id])
        return self._source(next(iter(history), None), selection)

    def _checked_review(self, source: Path, review_id: str) -> dict:
        review = self.review.get(review_id)
        if review['status'] != 'completed' or Path(review['source_path']) != source:
            raise ValueError('body_review_source_unavailable')
        return self.review.checked(review_id)

    @staticmethod
    def _review_error(exc: Exception, source: LibraryPartSource) -> ActionError:
        code = str(exc) if isinstance(exc, ValueError) and str(exc).startswith('body_review_') else 'body_review_unavailable'
        return ActionError(code, 409, generation_id=source.generation_id)

    @staticmethod
    def _snapshot(review: dict, part: Part | None = None) -> LibraryPartSnapshot:
        edits = review.get('edits') or {}
        excluded = set(edits.get('excluded', []))
        selected = set(edits.get('selected', []))
        candidates = []
        for raw in edits.get('candidates', review['candidates']):
            if raw['id'] in excluded or (part is not None and raw['label'] != part):
                continue
            candidates.append(CandidateBox.model_validate(raw))
        return LibraryPartSnapshot(
            review_id=review['id'], width=review['width'], height=review['height'],
            parts=review.get('parts', []), truncated=review.get('truncated', False), candidates=candidates,
            selected_candidate_ids=[candidate.id for candidate in candidates if candidate.id in selected],
        )

    def compare(self, request: LibraryPartsCompare, _request_id: str | None) -> ActionResult:
        ids = list(dict.fromkeys(source.generation_id for source in request.images))
        history = {row['id']: row for row in HistoryRepository.get_many_existing(self.db, ids)}
        results = []
        for selected in request.images:
            source = self._source(history.get(selected.generation_id), selected)
            try:
                review = self.review.latest(source)
                snapshot = self._snapshot(review, request.part) if review is not None else None
            except (OSError, ValueError, KeyError) as exc:
                raise ActionError('body_review_unavailable', 409, generation_id=selected.generation_id) from exc
            results.append(LibraryPartReviewItem(**selected.model_dump(), review=snapshot))
        return ActionResult(data=LibraryPartsResult(items=results).model_dump(mode='json'))

    def crop(self, request: LibraryPartCrop, _request_id: str | None) -> ActionResult:
        source = self._one_source(request)
        try:
            review = self._checked_review(source, request.review_id)
            snapshot = self._snapshot(review)
            candidate = next((box for box in snapshot.candidates if box.id == request.candidate_id), None)
            if candidate is None:
                raise ValueError('body_review_selection_invalid')
            # Use the same orientation-normalized snapshot and rectangle as the Library preview.
            with Image.open(Path(review['directory']) / 'source.png') as original:
                if original.size != (snapshot.width, snapshot.height):
                    raise ValueError('body_review_source_changed')
                x0, y0, x1, y1 = candidate.box
                region = original.crop((math.floor(x0 * original.width), math.floor(y0 * original.height),
                                        math.ceil(x1 * original.width), math.ceil(y1 * original.height)))
            stream = BytesIO()
            region.save(stream, format='PNG')
            data = stream.getvalue()
        except (OSError, ValueError, KeyError) as exc:
            raise self._review_error(exc, request) from exc
        result = LibraryPartCropResult(
            **request.model_dump(), part=candidate.label, width=region.width, height=region.height,
            sha256=sha256(data).hexdigest(), image=InlineImage(mime_type='image/png', data_base64=b64encode(data).decode('ascii')),
        )
        return ActionResult(data=result.model_dump(mode='json'))

    def _change(self, name: str, request: LibraryPartSource, request_id: str, prepare) -> ActionResult:
        arguments = canonical_operation_json(request.model_dump(mode='json'))
        try:
            lease = resource_admission.acquire((), operation_id=name, timeout_seconds=0)
        except ResourceAdmissionTimeoutError as exc:
            raise ActionError('maintenance_restore_busy', 409) from exc
        with lease, reference_mutation_coordinator.acquire_publisher():
            history = HistoryRepository(self.db)
            with history.publication() as connection:
                create_action_receipt_table(connection)
                previous = read_action_receipt(connection, request_id, name, arguments)
                if previous is not None:
                    if 'error' in previous:
                        error = previous['error']
                        raise ActionError(error['code'], 409, generation_id=request.generation_id)
                    return ActionResult(data=previous, replayed=True)
                try:
                    pending, perform = prepare(uuid4().hex)
                except ActionError:
                    raise
                except (OSError, ValueError, KeyError) as exc:
                    raise self._review_error(exc, request) from exc
                save_action_receipt(connection, request_id, name, arguments, pending.model_dump_json())

            def record(result: str) -> None:
                with history.publication() as connection:
                    connection.execute('update action_receipts set result_json=? where request_id=? and action=?',
                                       (result, request_id, name))

            try:
                completed: BaseModel = perform()
            except (OSError, ValueError, KeyError) as exc:
                error = self._review_error(exc, request)
                record(canonical_operation_json({'error': error.detail}))
                raise error from exc
            record(completed.model_dump_json())
            return ActionResult(data=completed.model_dump(mode='json'))

    def detect(self, request: LibraryPartDetect, request_id: str) -> ActionResult:
        def prepare(review_id: str):
            source = self._one_source(request)
            if not self.review.capabilities()['detector']:
                raise ValueError('body_review_model_unavailable')
            root = resolve_current_output_root(self.data_dir)
            pending = LibraryPartDetectionResult(generation_id=request.generation_id, artifact_id=request.artifact_id,
                                                 review_id=review_id, review=None, status='unconfirmed')

            def perform():
                row = self.review.create(source, root, review_id)
                row['parts'] = request.parts
                review = self.review.run(row, 'detect', review_id)
                return pending.model_copy(update={'status': 'completed', 'review': self._snapshot(review)})

            return pending, perform

        return self._change('library.parts.detect', request, request_id, prepare)

    def prepare_mask(self, request: LibraryPartMaskPrepare, request_id: str) -> ActionResult:
        def prepare(operation_id: str):
            source = self._one_source(request)
            row = self._checked_review(source, request.review_id)
            wanted = set(request.candidate_ids)
            candidates = [candidate.model_dump() for candidate in self._snapshot(row).candidates if candidate.id in wanted]
            if len(candidates) != len(wanted):
                raise ValueError('body_review_selection_invalid')
            reuse = candidates == row.get('selection') and bool(row['masks'])
            if not reuse and not self.review.capabilities()['sam']:
                raise ValueError('body_review_model_unavailable')
            pending = LibraryPartMaskResult(generation_id=request.generation_id, artifact_id=request.artifact_id,
                                            review_id=request.review_id, status='unconfirmed')

            def perform():
                # Recheck after admission; use a private copy in the ordinary mask assets store.
                current = self._checked_review(source, request.review_id)
                allowed = [item.model_dump() for item in self._snapshot(current).candidates if item.id in wanted]
                if allowed != candidates:
                    raise ValueError('body_review_selection_changed')
                if not reuse:
                    current = self.review.run(current, 'mask', operation_id, candidates)
                content = mask_union_png(current, candidates)
                mask, _ = ActionImageRepository(self.data_dir).save(
                    ImageUploadMetadata(filename=f'body-review-{request.review_id}.png', role='mask'),
                    f'body-review:{operation_id}', content,
                )
                return pending.model_copy(update={'status': 'completed', 'init_image_path': str(source),
                                                  'mask': mask, 'reused_existing_mask': reuse})

            return pending, perform

        return self._change('library.parts.mask.prepare', request, request_id, prepare)
