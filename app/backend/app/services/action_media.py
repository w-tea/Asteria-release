"""Expose existing Video Studio and Music Studio operations through core actions."""
from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException

from app.core.config import AppConfig
from app.schemas.action_contracts import (
    ActionResult, AssistantDraftRecord, DraftList, GenerationJobGet, GenerationStart, MediaArtifactReference,
    MediaCapabilities, MediaCapabilityGet, MediaSetupStatus, MusicDraftValidation,
    MediaGenerationJob, MediaGenerationRetry, MediaGenerationStart, MediaHistoryGet,
    MediaHistoryResult, MusicGenerationDraft, RestoreMediaDraft, SaveMediaDraft, VideoGenerationDraft,
    VideoUpscaleSourceGet, VideoUpscaleSourceSummary,
)
from app.schemas.media import MediaArtifact
from app.schemas.media_model_catalog import MediaModelCatalogResponse, MediaModelDomain
from app.schemas.music import MusicRequest
from app.schemas.operations import OperationRecord
from app.schemas.video import VideoRequest
from app.schemas.video_upscale import VideoUpscaleRequest
from app.services.action_dispatch import ActionError, _Action
from app.services.action_receipts import ActionRequestIdentity
from app.services.action_media_sources import MEDIA_IMPORT_SCHEMA
from app.services.assistant_drafts import AssistantDraftRepository, DraftConflict
from app.services.music_jobs import MUSIC_OPERATION_KIND
from app.services.music_local_adapter import MusicLocalAdapterError
from app.services.music_packet import MusicPacketError
from app.services.operation_repository import OperationNotFoundError, OperationRepository, OperationRetryError
from app.services.video_executor import VIDEO_OPERATION_KIND, VideoOperationService
from app.services.video_local_adapter import VideoLocalAdapterError
from app.services.video_packet import VideoPacketError
from app.services.video_upscale_jobs import VIDEO_UPSCALE_OPERATION_KIND, VideoUpscaleJobError


_JOB_SERVICES = {VIDEO_OPERATION_KIND: 'video', MUSIC_OPERATION_KIND: 'audio', VIDEO_UPSCALE_OPERATION_KIND: 'upscale'}
_DRAFT_MODES = {'video': 'video', 'audio': 'music'}
_DRAFT_TYPES = {'video': VideoGenerationDraft, 'audio': MusicGenerationDraft}
_REQUEST_TYPES = {'video': VideoRequest, 'audio': MusicRequest}
_ARTIFACT_KINDS = {'video', 'audio', 'poster'}


class MediaActions:
    def __init__(self, config: AppConfig, drafts: Callable[[], AssistantDraftRepository]):
        self.config = config
        self.drafts = drafts

    @property
    def db_path(self):
        return self.config.data_dir / 'asteria.db'

    def catalog(self):
        from app.api.music import MusicModelBundleListResponse
        from app.api.video_upscale import VideoUpscaleJobSubmission

        return {
            'media.capabilities.get': _Action(
                'Read available video or music modes and setup status',
                'Describe the existing Studio modes and managed setup status. This does not install or start anything.',
                MediaCapabilityGet, MediaCapabilities, True, self.capabilities),
            'media.models.list': _Action(
                'List installed Video Studio or Music Studio model assets',
                'Read existing component and LoRA identities and compatibility from the Studio model catalog.',
                MediaCapabilityGet, MediaModelCatalogResponse, True, self.models),
            'music.model_bundles.list': _Action(
                'List Music Studio model bundles',
                'Read the registered Music model bundle identities for MusicRequest. This does not select a bundle.',
                DraftList, MusicModelBundleListResponse, True, self.music_bundles),
            'music.drafts.validate': _Action(
                'Check a saved music draft',
                'Use Music Studio preflight to check a saved revision without generating or changing it.',
                GenerationStart, MusicDraftValidation, True, self.validate_music),
            'media.drafts.save': _Action(
                'Save video or music settings as a named draft',
                'Use the existing VideoRequest or MusicRequest with registered source artifact IDs. '
                'Omit existing to create a new draft. To update, give the intended draft ID and current revision. '
                'This does not start generation, download models or replace unsaved settings in a Studio.',
                SaveMediaDraft, AssistantDraftRecord, False, self.save_draft),
            'media.start': _Action(
                'Generate video or music from saved settings',
                'Start one Video Studio or Music Studio job from the exact saved draft revision. '
                'Use kind video for a video draft, audio for a music draft. A separate request starts a separate job. '
                'Video sequences use their own operations, not this single generation entry.',
                MediaGenerationStart, MediaGenerationJob, False, self.start),
            'media.jobs.get': _Action(
                'Read video or music progress and results',
                'Read the selected video, video upscale or music job. Use media.history.get with a returned generation ID to locate saved media.',
                GenerationJobGet, MediaGenerationJob, True, self.get_job),
            'media.jobs.cancel': _Action(
                'Stop a selected video or music job',
                'Ask the existing media job to stop. This does not stop the AI answer, any other job, or remove completed media.',
                GenerationJobGet, MediaGenerationJob, False, self.cancel),
            'media.jobs.retry': _Action(
                'Retry a selected video or music job',
                'Retry with the Studio’s saved inputs and retain completed candidates. '
                'Reuse the same request_id only when resending this retry.',
                MediaGenerationRetry, MediaGenerationJob, False, self.retry),
            'media.history.get': _Action(
                'Read saved video or music settings and artifact references',
                'Read the registered result and settings, including the exact Seed as decimal text. '
                'Returns relative Asteria artifact URLs, not file paths or embedded media.',
                MediaHistoryGet, MediaHistoryResult, True, self.history),
            'media.history.restore_draft': _Action(
                'Save a new video or music draft from history',
                'Use the recorded settings and existing source IDs. The original remains unchanged. '
                'This only saves a draft and never queues a job.',
                RestoreMediaDraft, AssistantDraftRecord, False, self.restore),
            'video.upscale.source.get': _Action(
                'Check the video selected for upscaling',
                'Check whether the registered video has the required working data. Returns source and default target details without starting a job.',
                VideoUpscaleSourceGet, VideoUpscaleSourceSummary, True, self.upscale_source),
            'video.upscale.start': _Action(
                'Upscale a selected saved video',
                'Use video.upscale.source.get to confirm the source identity and target information. '
                'Pass the same preview confirmation and settings as Video Studio. Use media.jobs.get to follow the result.',
                VideoUpscaleJobSubmission, MediaGenerationJob, False, self.upscale_start),
        }

    def _service(self, kind: str):
        from app.api import music, video, video_upscale

        # The same process-owned services as the Studios; no fallback constructor
        # that could initialize another environment's database or start a runtime.
        service = {'video': video._video_service, 'audio': music._music_service, 'upscale': video_upscale._video_upscale_service}[kind]
        db = self.db_path.resolve()
        if service is None or service.history.db_path.resolve() != db or service.coordinator.repository.db_path.resolve() != db:
            raise ActionError('action_runtime_configuration_mismatch', 503)
        return service

    def _operation(self, job_id: str) -> OperationRecord:
        try:
            operation = OperationRepository(self.db_path).get(job_id)
        except OperationNotFoundError as exc:
            raise ActionError('media_job_not_found', 404) from exc
        if operation.kind not in _JOB_SERVICES:
            raise ActionError('media_job_not_found', 404)
        return operation

    def capabilities(self, request: MediaCapabilityGet, _request_id: str | None) -> ActionResult:
        from app.api.music import music_capabilities
        from app.api.music_setup import MusicSetupError, get_music_setup_service
        from app.api.video import video_capabilities
        from app.api.video_setup import VideoSetupError, get_video_setup_service

        try:
            if request.kind == 'video':
                providers = video_capabilities().providers
                service = get_video_setup_service(self.config)
                plans = [(feature, service.plan(feature)) for feature in ('base', 'reference')]
            else:
                providers = music_capabilities().providers
                service = get_music_setup_service(self.config)
                profile = service.selected_execution_profile()
                plans = [(profile, service.plan(profile))]
        except (VideoSetupError, MusicSetupError) as exc:
            raise ActionError(exc.code, 409) from exc
        result = MediaCapabilities(providers=providers, setup=tuple(
            MediaSetupStatus(feature=feature, state=plan.state, reason_codes=plan.reason_codes)
            for feature, plan in plans
        ))
        return ActionResult(data=result.model_dump(mode='json'))

    def models(self, request: MediaCapabilityGet, _request_id: str | None) -> ActionResult:
        from app.api.models import read_media_catalog

        domain = MediaModelDomain.VIDEO if request.kind == 'video' else MediaModelDomain.MUSIC
        return ActionResult(data=read_media_catalog(self.config, domain).model_dump(mode='json'))

    def music_bundles(self, _request: DraftList, _request_id: str | None) -> ActionResult:
        from app.api.music import read_music_model_bundles

        return ActionResult(data=read_music_model_bundles(self.config).model_dump(mode='json'))

    def _draft(self, request: GenerationStart, kind: str):
        draft = self.drafts().get(request.draft_id)
        if draft.revision != request.expected_revision:
            raise DraftConflict('assistant_draft_revision_conflict', draft)
        if draft.document.get('mode') != _DRAFT_MODES[kind]:
            raise ActionError('assistant_draft_kind_mismatch', 422)
        return draft, _DRAFT_TYPES[kind].model_validate(draft.document)

    def validate_music(self, request: GenerationStart, _request_id: str | None) -> ActionResult:
        _, document = self._draft(request, 'audio')
        try:
            checked = self._service('audio').draft(document.input)
        except (MusicLocalAdapterError, MusicPacketError) as exc:
            raise ActionError(exc.code, 422) from exc
        return ActionResult(data=MusicDraftValidation.model_validate(checked).model_dump(mode='json'))

    def save_draft(self, request: SaveMediaDraft, request_id: str) -> ActionResult:
        drafts = self.drafts()
        document = request.document.model_dump(mode='json')
        action = 'media.drafts.save'
        if request.existing is None:
            record, replayed = drafts.create(request_id, request.title, document, action=action)
        else:
            record, replayed = drafts.update(request_id, request.existing.draft_id, request.existing.expected_revision,
                                             request.title, document, action=action)
        return ActionResult(data=record.action_data(), replayed=replayed)

    @staticmethod
    def _result(operation: OperationRecord, replayed: bool = False) -> ActionResult:
        generation_ids = VideoOperationService._snapshot_generation_ids(operation.request_snapshot)
        job = MediaGenerationJob.model_validate({
            **operation.model_dump(exclude={'request_snapshot'}),
            'generation_id': generation_ids[0], 'generation_ids': generation_ids,
        })
        return ActionResult(data=job.model_dump(mode='json'), replayed=replayed)

    def start(self, request: MediaGenerationStart, request_id: str) -> ActionResult:
        # Resolve a redelivery before observing the draft's possibly newer version.
        arguments = request.model_dump(mode='json')
        identity = ActionRequestIdentity(request_id, 'media.start', arguments)
        accepted = OperationRepository(self.db_path).get_action_operation(request_id, identity.action, arguments)
        if accepted is not None:
            return self._result(accepted, True)
        draft, document = self._draft(request, request.kind)
        service = self._service(request.kind)
        try:
            submitted = service.submit(
                document.input, source_actor='assistant', action_request=identity,
                draft={'id': draft.id, 'revision': draft.revision},
                resource_authorization=request.resource_authorization, run_anyway=request.run_anyway is True,
            )
        except (VideoLocalAdapterError, VideoPacketError, MusicLocalAdapterError, MusicPacketError) as exc:
            raise ActionError(exc.code, 409 if exc.code.endswith('_local_setup_required') else 422) from exc
        return self._result(submitted.operation, submitted.replayed)

    def get_job(self, request: GenerationJobGet, _request_id: str | None) -> ActionResult:
        return self._result(self._operation(request.job_id))

    def cancel(self, request: GenerationJobGet, request_id: str) -> ActionResult:
        operation = self._operation(request.job_id)
        service = self._service(_JOB_SERVICES[operation.kind])
        try:
            target, created = service.coordinator.repository.bind_action_operation(
                request_id, 'media.jobs.cancel', request.model_dump(mode='json'), operation.operation_id,
            )
            return self._result(service.coordinator.request_cancel(target.operation_id), not created)
        except OperationNotFoundError as exc:
            raise ActionError('media_job_not_found', 404) from exc

    def retry(self, request: MediaGenerationRetry, request_id: str) -> ActionResult:
        from app.api.video_upscale import _status_for

        arguments = request.model_dump(mode='json')
        identity = ActionRequestIdentity(request_id, 'media.jobs.retry', arguments)
        accepted = OperationRepository(self.db_path).get_action_operation(request_id, identity.action, arguments)
        if accepted is not None:
            return self._result(accepted, True)
        operation = self._operation(request.job_id)
        service = self._service(_JOB_SERVICES[operation.kind])
        try:
            retry, replayed = service.retry_with_receipt(
                request.job_id, source_actor='assistant', action_request=identity,
                resource_authorization=request.resource_authorization, run_anyway=request.run_anyway is True,
            )
        except OperationNotFoundError as exc:
            raise ActionError('media_job_not_found', 404) from exc
        except OperationRetryError as exc:
            raise ActionError(str(exc), 409) from exc
        except VideoUpscaleJobError as exc:
            raise ActionError(exc.code, _status_for(exc)) from exc
        except VideoLocalAdapterError as exc:
            status = 409 if operation.kind == VIDEO_OPERATION_KIND and exc.code == 'video_local_setup_required' else 422
            raise ActionError(exc.code, status) from exc
        except MusicLocalAdapterError as exc:
            raise ActionError(exc.code, 409) from exc
        return self._result(retry, replayed)

    def upscale_source(self, request: VideoUpscaleSourceGet, _request_id: str | None) -> ActionResult:
        from app.api.video_upscale import read_video_upscale_source

        try:
            source = read_video_upscale_source(self._service('upscale'), request.generation_id, request.artifact_id)
        except HTTPException as exc:
            raise ActionError(exc.detail['code'], exc.status_code) from exc
        return ActionResult(data=VideoUpscaleSourceSummary.model_validate(source).model_dump(mode='json'))

    def upscale_start(self, submission, request_id: str) -> ActionResult:
        from app.api.video_upscale import _status_for

        identity = ActionRequestIdentity(request_id, 'video.upscale.start', submission.model_dump(mode='json'))
        try:
            accepted = self._service('upscale').submit(
                submission.request, source_actor='assistant', action_request=identity,
                resource_authorization=submission.resource_authorization, run_anyway=submission.run_anyway is True,
            )
        except VideoUpscaleJobError as exc:
            raise ActionError(exc.code, _status_for(exc)) from exc
        except VideoLocalAdapterError as exc:
            raise ActionError(exc.code, 422) from exc
        return self._result(accepted.operation, accepted.replayed)

    def _history(self, request: MediaHistoryGet, *, for_restore: bool = False) -> MediaHistoryResult:
        # Use the already-open Studio repository. Its constructor migrates old
        # rows, so constructing another one would make a result read a write.
        item = self._service(request.kind).history.get(request.generation_id)
        prefix = 'video' if request.kind == 'video' else 'music'
        if item is None or item.get('media_kind') != request.kind or (
            for_restore and item.get('status') not in {'completed', 'completed_with_warnings'}
        ):
            raise ActionError(f'{prefix}_parameters_not_found', 404)
        settings = item.get('settings')
        snapshot = settings.get('request_snapshot') if isinstance(settings, dict) else None
        upscale = (request.kind == 'video' and isinstance(settings, dict)
                   and settings.get('schema_version') == 'asteria.video-upscale.history.v1')
        assembly = (request.kind == 'video' and isinstance(settings, dict)
                    and settings.get('schema_version') == 'asteria.video.sequence.assembly.history.v1')
        imported = isinstance(settings, dict) and settings.get('schema_version') == MEDIA_IMPORT_SCHEMA
        try:
            if (assembly or imported) and not for_restore:
                inputs, seed_decimal = None, None
            elif upscale and not for_restore:
                inputs = VideoUpscaleRequest.model_validate(snapshot)
                seed_decimal = str(int(settings['resolved_seed']))
            else:
                inputs = _REQUEST_TYPES[request.kind].model_validate(snapshot)
                seed_decimal = str(inputs.seed)
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionError(f'{prefix}_parameters_unavailable', 409) from exc
        artifacts = []
        for raw in item.get('artifacts', []):
            try:
                artifact = MediaArtifact.model_validate(raw)
            except ValueError:
                continue
            if artifact.kind in _ARTIFACT_KINDS:
                artifacts.append(MediaArtifactReference.model_validate({
                    **artifact.model_dump(exclude={'path', 'adapter_metadata'}),
                    'url': f'/api/generations/history/{request.generation_id}/artifacts/{artifact.artifact_id}',
                }))
        return MediaHistoryResult(generation_id=request.generation_id, kind=request.kind, status=item['status'],
                                  seed_decimal=seed_decimal, request=inputs, artifacts=tuple(artifacts),
                                  sequence_id=settings.get('sequence_id') if assembly else None)

    def history(self, request: MediaHistoryGet, _request_id: str | None) -> ActionResult:
        return ActionResult(data=self._history(request).action_data())

    def restore(self, request: RestoreMediaDraft, request_id: str) -> ActionResult:
        def document():
            original = self._history(request, for_restore=True)
            return _DRAFT_TYPES[request.kind](mode=_DRAFT_MODES[request.kind], input=original.request).model_dump(mode='json')

        record, replayed = self.drafts().restore_media_history(
            request_id, request.kind, request.generation_id, request.title, document,
        )
        return ActionResult(data=record.action_data(), replayed=replayed)
