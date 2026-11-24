"""Assistant/MCP connection to the existing Video Sequence state machine."""
from __future__ import annotations

from hashlib import sha256
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.core.config import AppConfig
from app.schemas.action_contracts import ActionInput, ActionResult, GenerationStart, MediaResourceChoice, VideoGenerationDraft
from app.schemas.operations import OperationRecord, OperationStatus
from app.schemas.video import VideoRequest
from app.schemas.video_sequence import VideoSequence, VideoSequenceSegmentStatus, VideoSequenceStatus
from app.services.action_dispatch import ActionError, _Action
from app.services.action_receipts import ActionRequestIdentity
from app.services.assistant_drafts import AssistantDraftRepository, DraftConflict
from app.services.operation_repository import OperationNotFoundError
from app.services.video_local_adapter import VideoLocalAdapterError
from app.services.video_packet import VideoPacketError
from app.services.video_sequence_context import VideoSequenceContextError
from app.services.video_sequence_jobs import VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND, VIDEO_SEQUENCE_CLIP_OPERATION_KIND
from app.services.video_sequence_repository import VideoSequenceRepositoryError


class SequenceGet(ActionInput):
    sequence_id: str = Field(min_length=1)


class SequenceSegmentTarget(SequenceGet):
    segment_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=1, strict=True)
    expected_segment_revision: int = Field(ge=1, strict=True)


class SequenceClipStart(SequenceSegmentTarget, MediaResourceChoice):
    retry: bool = False


class SequenceClipCancel(SequenceSegmentTarget):
    job_id: str = Field(min_length=1)


class SequenceCandidate(SequenceSegmentTarget):
    candidate_artifact_id: str = Field(min_length=1)


class SequenceContinue(SequenceCandidate):
    next_draft: GenerationStart


class SequenceRegeneration(SequenceSegmentTarget):
    draft: GenerationStart
    parent_segment_revision: int = Field(ge=1, strict=True)


class SequenceAssemblyRetry(SequenceGet):
    expected_revision: int = Field(ge=1, strict=True)


class SequenceJobGet(SequenceGet):
    job_id: str = Field(min_length=1)


class SegmentSummary(BaseModel):
    segment_id: str
    ordinal: int
    revision: int
    parent_segment_revision: int | None
    status: VideoSequenceSegmentStatus
    request_snapshot: VideoRequest
    candidate_artifact_ids: tuple[str, ...]
    candidate_generation_ids: dict[str, int]
    accepted_candidate_id: str | None
    accepted_revision: int | None
    operation_id: str | None


class SequenceSummary(BaseModel):
    sequence_id: str
    status: VideoSequenceStatus
    revision: int
    active_chain_revision: int
    segments: tuple[SegmentSummary, ...]
    active_segment_ids: tuple[str, ...]
    assembly_operation_id: str | None
    assembly_generation_id: int | None
    assembled_artifact_id: str | None


class SequenceJobSummary(BaseModel):
    operation_id: str
    kind: Literal['video.sequence.clip', 'video.sequence.assemble']
    generation_id: int
    status: OperationStatus
    phase: str
    progress_current: int
    progress_total: int | None
    progress_unit: str
    artifact_refs: tuple[str, ...]
    error_code: str | None
    retryable: bool


class SequenceClipResult(BaseModel):
    sequence: SequenceSummary
    operation: SequenceJobSummary


class SequenceActions:
    def __init__(self, config: AppConfig):
        self.config = config

    def catalog(self):
        return {
            'video.sequences.create': _Action(
                'Create a video sequence from a saved revision',
                'Use a video draft with VideoRequest mode sequence. This creates the first draft clip only; it does not generate.',
                GenerationStart, SequenceSummary, False, self.create),
            'video.sequences.get': _Action(
                'Read a video sequence and its candidates',
                'Read saved clip revisions, candidate history and artifact IDs and accepted clips. A candidate is not yet adopted.',
                SequenceGet, SequenceSummary, True, self.get),
            'video.sequences.clips.start': _Action(
                'Generate a candidate for a selected sequence clip',
                'Use the current sequence and clip revisions. Set retry for the existing retry procedure. '
                'Generation never adopts the candidate or assembles a finished video.',
                SequenceClipStart, SequenceClipResult, False, self.start),
            'video.sequences.clips.cancel': _Action(
                'Stop the specified sequence clip job',
                'The sequence, clip, revision and active job must match. No other sequence or already adopted clip is changed.',
                SequenceClipCancel, SequenceJobSummary, False, self.cancel),
            'video.sequences.clips.resume': _Action(
                'Return to an interrupted sequence clip',
                'Use the existing resume transition. This does not start a model job; request a retry separately when needed.',
                SequenceSegmentTarget, SequenceSummary, False, self.resume),
            'video.sequences.clips.accept_continue': _Action(
                'Adopt the chosen clip and prepare the next one',
                'Use the user-selected candidate and a saved revision for the next clip. This does not start the next clip.',
                SequenceContinue, SequenceSummary, False, self.accept_continue),
            'video.sequences.clips.accept_finish': _Action(
                'Adopt the chosen clip and start assembling the sequence',
                'Use the user-selected candidate to finish the sequence. An assembling state does not mean a saved video exists yet.',
                SequenceCandidate, SequenceClipResult, False, self.accept_finish),
            'video.sequences.clips.replace_accepted': _Action(
                'Replace the adopted clip with the candidate the user chose',
                'Use the current sequence and clip revisions. Later clips need regeneration; prior files stay in history. '
                'This does not generate clips or start assembly.',
                SequenceCandidate, SequenceSummary, False, self.replace_accepted),
            'video.sequences.clips.prepare_regeneration': _Action(
                'Prepare a downstream clip after replacing an adopted clip',
                'Use the current clip revision, its active predecessor’s accepted revision and a saved video draft. '
                'Use clips.start separately to generate.',
                SequenceRegeneration, SequenceSummary, False, self.prepare_regeneration),
            'video.sequences.assembly.retry': _Action(
                'Retry the failed video assembly',
                'Reuse the accepted clips and repeat the existing assembly step only; do not regenerate any clip.',
                SequenceAssemblyRetry, SequenceClipResult, False, self.retry_assembly),
            'video.sequences.jobs.get': _Action(
                'Read the selected sequence job',
                'Read progress and results for the job belonging to the specified sequence.',
                SequenceJobGet, SequenceJobSummary, True, self.get_job),
        }

    def _service(self):
        from app.api.video_sequences import _sequence_service

        db_path = (self.config.data_dir / 'asteria.db').resolve()
        if (_sequence_service is None or _sequence_service.sequences.db_path != db_path or
                _sequence_service.history.db_path.resolve() != db_path or
                _sequence_service.coordinator.repository.db_path.resolve() != db_path):
            raise ActionError('action_runtime_configuration_mismatch', 503)
        return _sequence_service

    @staticmethod
    def _error(error):
        from app.api.video_sequences import _raise_api_error

        try:
            _raise_api_error(error)
        except HTTPException as exc:
            raise ActionError(exc.detail['code'], exc.status_code) from error

    @staticmethod
    def _sequence(value: VideoSequence) -> SequenceSummary:
        return SequenceSummary.model_validate(value.model_dump(mode='json'))

    @staticmethod
    def _job(operation: OperationRecord) -> SequenceJobSummary:
        return SequenceJobSummary.model_validate({
            **operation.model_dump(exclude={'request_snapshot'}),
            'generation_id': operation.request_snapshot.get('generation_id'),
        })

    @staticmethod
    def _receipt_id(request_id: str) -> str:
        return 'action-' + sha256(request_id.encode('utf-8')).hexdigest()

    @staticmethod
    def _draft_input(drafts: AssistantDraftRepository, request: GenerationStart) -> VideoRequest:
        draft = drafts.get(request.draft_id)
        if draft.revision != request.expected_revision:
            raise DraftConflict('assistant_draft_revision_conflict', draft)
        if draft.document.get('mode') != 'video':
            raise ActionError('assistant_draft_kind_mismatch', 422)
        return VideoGenerationDraft.model_validate(draft.document).input

    def create(self, request: GenerationStart, request_id: str) -> ActionResult:
        service = self._service()
        drafts = AssistantDraftRepository(service.history.db_path)
        identity = ActionRequestIdentity(request_id, 'video.sequences.create', request.model_dump(mode='json'))

        def create_from_draft():
            return service.create(self._draft_input(drafts, request), receipt_id=self._receipt_id(request_id))

        try:
            result, replayed = service.sequences.write_action(identity, create_from_draft)
        except (VideoLocalAdapterError, VideoPacketError, VideoSequenceRepositoryError) as exc:
            self._error(exc)
        return ActionResult(data=self._sequence(result).model_dump(mode='json'), replayed=replayed)

    def get(self, request: SequenceGet, _request_id: str | None) -> ActionResult:
        try:
            value = self._service().get(request.sequence_id)
        except VideoSequenceRepositoryError as exc:
            self._error(exc)
        return ActionResult(data=self._sequence(value).model_dump(mode='json'))

    def start(self, request: SequenceClipStart, request_id: str) -> ActionResult:
        identity = ActionRequestIdentity(request_id, 'video.sequences.clips.start', request.model_dump(mode='json'))
        service = self._service()
        try:
            value = service.generate(
                request.sequence_id, segment_id=request.segment_id,
                expected_revision=request.expected_revision, expected_segment_revision=request.expected_segment_revision,
                resource_authorization=request.resource_authorization, run_anyway=request.run_anyway is True,
                retry=request.retry, source_actor='assistant', receipt_id=self._receipt_id(request_id), action_request=identity,
            )
        except (VideoLocalAdapterError, VideoPacketError, VideoSequenceContextError, VideoSequenceRepositoryError) as exc:
            # Another delivery can commit between preflight's receipt read and
            # the revision check. Return that exact operation, not a stale error.
            accepted = service.coordinator.repository.get_action_operation(request_id, identity.action, identity.arguments)
            if accepted is None:
                self._error(exc)
            value = service._clip_replay(request.sequence_id, accepted)
        return self._clip_result(value)

    def _clip_result(self, value) -> ActionResult:
        result = SequenceClipResult(sequence=self._sequence(value.sequence), operation=self._job(value.operation))
        return ActionResult(data=result.model_dump(mode='json'), replayed=value.replayed)

    def cancel(self, request: SequenceClipCancel, request_id: str) -> ActionResult:
        identity = ActionRequestIdentity(request_id, 'video.sequences.clips.cancel', request.model_dump(mode='json'))
        try:
            value, replayed = self._service().cancel_with_receipt(
                request.sequence_id, segment_id=request.segment_id, operation_id=request.job_id,
                expected_revision=request.expected_revision, expected_segment_revision=request.expected_segment_revision,
                receipt_id=self._receipt_id(request_id), action_request=identity,
            )
        except (OperationNotFoundError, VideoSequenceRepositoryError) as exc:
            self._error(exc)
        return ActionResult(data=self._job(value).model_dump(mode='json'), replayed=replayed)

    def resume(self, request: SequenceSegmentTarget, request_id: str) -> ActionResult:
        repository = self._service().sequences
        arguments = request.model_dump(mode='json')
        identity = ActionRequestIdentity(request_id, 'video.sequences.clips.resume', arguments)
        try:
            value, replayed = repository.write_action(identity, lambda: repository.resume(
                **arguments, receipt_id=self._receipt_id(request_id),
            ))
        except VideoSequenceRepositoryError as exc:
            self._error(exc)
        return ActionResult(data=self._sequence(value).model_dump(mode='json'), replayed=replayed)

    def accept_continue(self, request: SequenceContinue, request_id: str) -> ActionResult:
        service = self._service()
        drafts = AssistantDraftRepository(service.history.db_path)
        identity = ActionRequestIdentity(request_id, 'video.sequences.clips.accept_continue', request.model_dump(mode='json'))

        def update():
            return service.accept_and_continue(
                request.sequence_id, segment_id=request.segment_id, candidate_artifact_id=request.candidate_artifact_id,
                next_request=self._draft_input(drafts, request.next_draft), expected_revision=request.expected_revision,
                expected_segment_revision=request.expected_segment_revision, receipt_id=self._receipt_id(request_id),
            )

        try:
            value, replayed = service.sequences.write_action(identity, update)
        except (VideoLocalAdapterError, VideoSequenceRepositoryError) as exc:
            self._error(exc)
        return ActionResult(data=self._sequence(value).model_dump(mode='json'), replayed=replayed)

    def accept_finish(self, request: SequenceCandidate, request_id: str) -> ActionResult:
        identity = ActionRequestIdentity(request_id, 'video.sequences.clips.accept_finish', request.model_dump(mode='json'))
        try:
            value = self._service().accept_and_finish(
                request.sequence_id, segment_id=request.segment_id, candidate_artifact_id=request.candidate_artifact_id,
                expected_revision=request.expected_revision, expected_segment_revision=request.expected_segment_revision,
                receipt_id=self._receipt_id(request_id), source_actor='assistant', action_request=identity,
            )
        except (VideoLocalAdapterError, VideoSequenceRepositoryError) as exc:
            self._error(exc)
        return self._clip_result(value)

    def replace_accepted(self, request: SequenceCandidate, request_id: str) -> ActionResult:
        repository = self._service().sequences
        identity = ActionRequestIdentity(request_id, 'video.sequences.clips.replace_accepted', request.model_dump(mode='json'))
        try:
            value, replayed = repository.write_action(identity, lambda: repository.replace_accepted(
                **request.model_dump(), receipt_id=self._receipt_id(request_id),
            ))
        except VideoSequenceRepositoryError as exc:
            self._error(exc)
        return ActionResult(data=self._sequence(value).model_dump(mode='json'), replayed=replayed)

    def prepare_regeneration(self, request: SequenceRegeneration, request_id: str) -> ActionResult:
        service = self._service()
        drafts = AssistantDraftRepository(service.history.db_path)
        identity = ActionRequestIdentity(request_id, 'video.sequences.clips.prepare_regeneration', request.model_dump(mode='json'))
        try:
            value, replayed = service.sequences.write_action(identity, lambda: service.sequences.prepare_regeneration(
                **request.model_dump(exclude={'draft'}), request_snapshot=self._draft_input(drafts, request.draft),
                receipt_id=self._receipt_id(request_id),
            ))
        except (VideoLocalAdapterError, VideoSequenceRepositoryError) as exc:
            self._error(exc)
        return ActionResult(data=self._sequence(value).model_dump(mode='json'), replayed=replayed)

    def retry_assembly(self, request: SequenceAssemblyRetry, request_id: str) -> ActionResult:
        identity = ActionRequestIdentity(request_id, 'video.sequences.assembly.retry', request.model_dump(mode='json'))
        try:
            value = self._service().retry_assembly(
                request.sequence_id, expected_revision=request.expected_revision,
                receipt_id=self._receipt_id(request_id), source_actor='assistant', action_request=identity,
            )
        except VideoSequenceRepositoryError as exc:
            self._error(exc)
        return self._clip_result(value)

    @staticmethod
    def job_sequence_id(operation: OperationRecord) -> str | None:
        if operation.kind == VIDEO_SEQUENCE_CLIP_OPERATION_KIND:
            context = operation.request_snapshot.get('sequence')
            owner = context.get('sequence_id') if isinstance(context, dict) else None
        elif operation.kind == VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND:
            owner = operation.request_snapshot.get('sequence_id')
        else:
            return None
        return owner if isinstance(owner, str) and owner else None

    def get_job(self, request: SequenceJobGet, _request_id: str | None) -> ActionResult:
        try:
            value = self._service().coordinator.get(request.job_id)
            if self.job_sequence_id(value) != request.sequence_id:
                raise OperationNotFoundError('operation_not_found')
        except OperationNotFoundError as exc:
            self._error(exc)
        return ActionResult(data=self._job(value).model_dump(mode='json'))
