"""Core-owned operations, deliberately independent from LLM/MCP transports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, TypeAdapter

from app.core.config import AppConfig
from app.schemas.action_contracts import (
    ActionCall, ActionDescription, ActionResult, AssistantDraftList, AssistantDraftRecord,
    DraftCreate, DraftDerive, DraftGet, DraftImageGet, DraftImageContent, DraftList, DraftRevisionGet, DraftUpdate,
    GenerationStart, GenerationJobGet, GenerationDraftDocument, GenerationComparisonGet, PrepareGenerationComparison,
    GenerationDraftValidation, JobMedia, JobsList, JobsListResult, ListedJob,
    ModelLibraryQuery, GenerationSchemaQuery, GenerationPresetQuery, SavedGenerationPresetList,
    ImageGet, ActionImageContent,
    LibrarySearch, LibrarySearchItem, LibrarySearchResult, LibraryCurationUpdate, LibraryCurationResult,
    LibraryBulkCurationUpdate, LibraryBulkCurationResult,
    LibraryComparisonGet, LibraryComparisonResult,
    LibraryExportResult,
    LibraryDelete, LibraryDeleteResult,
    GenerationHistoryGet, RestoreGenerationDraft, GenerationImageGet, GenerationImageContent,
    GenerationBranchStart, GenerationBranchDetails,
    GenerationXYZStart, GenerationXYZStatus,
)
from app.services.assistant_drafts import AssistantDraftRepository, DraftConflict, DraftNotFound, DraftRevisionNotFound
from app.services.action_receipts import ActionRequestConflict, ActionRequestIdentity
from app.schemas.prompt_comparison import PromptComparisonRequest, PromptComparisonStatus
from app.schemas.generation import XYZPlotRequest
from app.schemas.generation_repeat import (
    GenerationRepeatRecord, RepeatStart, RepeatGet, RepeatUpdate, RepeatStop,
    RepeatInputUpdate, RepeatStopRequest,
)
from app.services.operation_repository import OperationConflictError, OperationNotFoundError, OperationRepository


_IMAGE_JOB_KINDS = frozenset({'generation.txt2img', 'generation.img2img', 'generation.inpaint', 'generation.remaster'})


class ActionError(ValueError):
    def __init__(self, code: str, status_code: int, **detail):
        super().__init__(code)
        self.status_code = status_code
        self.detail = {'code': code, **detail}


@dataclass(frozen=True)
class _Action:
    title: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    read_only: bool
    run: Callable[[BaseModel, str | None], ActionResult]


class ActionDispatcher:
    def __init__(self, config: AppConfig):
        from app.api.generation import GenerationJobResponse, GenerationSchemaResponse
        from app.api.models import ModelProfileListResponse, ModelLibraryResponse
        from app.schemas.generation import GenerationFolderResponse, GenerationHistoryItem
        from app.schemas.library_body_review import (
            LibraryPartCrop, LibraryPartCropResult, LibraryPartDetect, LibraryPartDetectionResult,
            LibraryPartMaskPrepare, LibraryPartMaskResult, LibraryPartsCompare, LibraryPartsResult,
        )
        from app.services.action_media import MediaActions
        from app.services.action_video_sequence import SequenceActions
        from app.services.library_body_review import LibraryBodyReviews
        from app.services.library_delivery import LibraryDelivery

        self.config = config
        parts = LibraryBodyReviews(config)
        self._actions = {
            'images.get': _Action(
                'Read a saved input image or mask',
                'Read the bytes of an image previously saved in Asteria by its asset_id. '
                'This does not read arbitrary files on the host.',
                ImageGet, ActionImageContent, True, self._image),
            'models.profiles.list': _Action(
                'List model profiles',
                'Read Asteria model families, required component slots and prompt notation support. '
                'This describes capabilities, not whether models are installed.',
                DraftList, ModelProfileListResponse, True, self._profiles),
            'models.library.list': _Action(
                'List installed models and LoRAs',
                'Read the existing model library. In model_slot_overrides use the returned full path, '
                'not name or an invented relative path. Use returned identities instead of guessing filenames. '
                'This does not change the selected model or download anything.',
                ModelLibraryQuery, ModelLibraryResponse, True, self._library),
            'generation.schema.get': _Action(
                'Read generation fields and defaults',
                'Read the same per-profile fields and defaults shown in Generate. '
                'UI range hints are not extra restrictions; the core request schema validates input.',
                GenerationSchemaQuery, GenerationSchemaResponse, True, self._schema),
            'generation.presets.list': _Action(
                'Read saved generation inputs',
                'Find saved generation presets, prompt presets, advanced styles, or wildcard names. '
                'For a matching ID, read its saved values. Generation presets use the Generate screen format; '
                'use generation.schema.get for request fields. Reading does not apply or change settings.',
                GenerationPresetQuery, SavedGenerationPresetList, True, self._presets),
            'jobs.list': _Action(
                'Find image, video and music jobs',
                'Read existing jobs from Asteria, with optional media or status filters. '
                'Active jobs appear first. Use detail_action and detail_arguments to inspect a selected job. '
                'This does not start, stop, or modify any job.',
                JobsList, JobsListResult, True, self._list_jobs),
            'generation.drafts.create': _Action(
                'Save a generation draft',
                'Save a named image draft without generating or replacing another screen\'s unsaved input. '
                'Use media.drafts.save for video or music. '
                'Reuse request_id only when resending this same request.',
                DraftCreate, AssistantDraftRecord, False, self._create),
            'generation.drafts.update': _Action(
                'Update a generation draft',
                'Replace the specified saved draft using its current revision. '
                'On a revision conflict, read the current draft before deciding which changes to keep.',
                DraftUpdate, AssistantDraftRecord, False, self._update),
            'generation.drafts.derive': _Action(
                'Copy a saved draft with selected changes',
                'Create a separate draft from an exact saved revision, changing only input_changes. '
                'Use this for follow-up edits to keep the source image, mask, Seed, model, LoRAs and other '
                'settings unless the user asks to change them. The source remains unchanged. '
                'This saves only. Start an image with generation.start, or video/music with media.start.',
                DraftDerive, AssistantDraftRecord, False, self._derive),
            'generation.drafts.get': _Action(
                'Read a generation draft', 'Read a draft saved in Asteria by its ID.',
                DraftGet, AssistantDraftRecord, True, self._get),
            'generation.drafts.validate': _Action(
                'Check a saved image generation or edit draft',
                'Check a specific saved revision, including mode, required model slots, requested size and '
                'inspectable source images and masks. This does not start a job or guarantee successful execution. '
                'Unverified inputs and differences in image size are not rejected by this tool.',
                DraftRevisionGet, GenerationDraftValidation, True, self._validate_draft),
            'generation.drafts.images.get': _Action(
                'Read an image or mask from a saved editor revision',
                'Read actual image bytes using draft_id, revision and role (init or mask). '
                'Use the revision from the saved edit. Older saved versions remain distinct after cropping '
                'or later edits. This does not read unsaved client edits or arbitrary host files.',
                DraftImageGet, DraftImageContent, True, self._draft_image),
            'generation.drafts.list': _Action(
                'List generation drafts', 'Find named drafts saved in Asteria; unsaved client inputs are not included.',
                DraftList, AssistantDraftList, True, self._list),
            'generation.start': _Action(
                'Generate an image from a saved draft',
                'Start image generation from the specified saved revision. For video and music use media.start. '
                'A new request_id starts a new job, '
                'even for identical settings. Resend the same ID only to recover the original job.',
                GenerationStart, GenerationJobResponse, False, self._start),
            'generation.jobs.get': _Action(
                'Read image generation progress and results',
                'Read the current state, errors and results of the specified image generation job.',
                GenerationJobGet, GenerationJobResponse, True, self._job),
            'generation.jobs.cancel': _Action(
                'Stop a specified image generation',
                'Request cancellation of the specified image job through Asteria. '
                'This does not stop the AI answer or unrelated jobs. Completed results are kept. '
                'A cancelling response means the core is still stopping; read this same job for its final state.',
                GenerationJobGet, GenerationJobResponse, False, self._cancel_job),
            'generation.jobs.retry': _Action(
                'Retry the selected image generation',
                'Retry a failed or stopped image generation with the saved input. The original stays unchanged. '
                'A new request_id starts a separate retry; resend the same ID only to recover that retry.',
                GenerationJobGet, GenerationJobResponse, False, self._retry_job),
            'generation.comparisons.prepare': _Action(
                'Prepare a prompt comparison from a saved draft',
                'Freeze two conditions per Seed using Asteria prompt comparison. Saves only. '
                'The change can be a prompt fragment, Embedding, LoRA weight, prompt section or Advanced Style.',
                PrepareGenerationComparison, PromptComparisonStatus, False, self._prepare_comparison),
            'generation.comparisons.start': _Action(
                'Start the prepared prompt comparison',
                'Start both conditions through the normal comparison job. Resend the same ID only for redelivery.',
                GenerationComparisonGet, PromptComparisonStatus, False, self._start_comparison),
            'generation.comparisons.get': _Action(
                'Read prompt comparison progress and results',
                'Read states and history IDs for each condition. Use generation.images.get to read actual completed images.',
                GenerationComparisonGet, PromptComparisonStatus, True, self._get_comparison),
            'generation.comparisons.cancel': _Action(
                'Stop the specified prompt comparison',
                'Request cancellation through Asteria. Completed comparison images stay saved.',
                GenerationComparisonGet, PromptComparisonStatus, False, self._cancel_comparison),
            'generation.branches.get': _Action(
                'Read a recorded intermediate generation state',
                'Check the saved intermediate state for an image by history ID. Use only a registered saved point.',
                GenerationHistoryGet, GenerationBranchDetails, True, self._get_branch),
            'generation.branches.start': _Action(
                'Generate a branch from a saved intermediate state',
                'Use an existing registered state by history ID. Only prompt, negative_prompt or CFG may differ. '
                'Read progress using generation.jobs.get. This does not start sampling from the beginning.',
                GenerationBranchStart, GenerationJobResponse, False, self._start_branch),
            'generation.xyz.start': _Action(
                'Start an XYZ plot from a saved generation draft',
                'Use the existing XYZ axes and values and queue one plot. Read per-cell history IDs with generation.xyz.get.',
                GenerationXYZStart, GenerationXYZStatus, False, self._start_xyz),
            'generation.xyz.get': _Action(
                'Read XYZ progress and results',
                'Read the selected plot, completed cell history IDs, and aggregate result when available.',
                GenerationJobGet, GenerationXYZStatus, True, self._get_xyz),
            'generation.xyz.cancel': _Action(
                'Stop the selected XYZ plot',
                'Stops the indicated XYZ plot only; keeps cell results already saved.',
                GenerationJobGet, GenerationXYZStatus, False, self._cancel_xyz),
            'generation.repeats.start': _Action(
                'Start repeated image generation',
                'Repeat an exact saved draft revision in Asteria, even if this client disconnects. '
                'Zero limits mean unlimited; Asteria restart never automatically resumes a repeat. '
                'A new request_id creates a separate repeat. Read its current_job_id with generation.jobs.get.',
                RepeatStart, GenerationRepeatRecord, False, self._repeat_start),
            'generation.repeats.get': _Action(
                'Read a repeated generation',
                'Read counts, ending conditions, next input and the current/latest image job IDs.',
                RepeatGet, GenerationRepeatRecord, True, self._repeat_get),
            'generation.repeats.update': _Action(
                'Change the next repeated generation input',
                'Update next input using input_revision. Already accepted images and the source draft stay unchanged. '
                'On a conflict read the current repeat before choosing which changes to keep.',
                RepeatUpdate, GenerationRepeatRecord, False, self._repeat_update),
            'generation.repeats.stop': _Action(
                'Stop a specified repeated generation',
                'Prevent further images. stop_current=false lets the accepted image finish; true also cancels '
                'that image. Other repeats and finished results are kept. Read the repeat until it stops.',
                RepeatStop, GenerationRepeatRecord, False, self._repeat_stop),
            'library.search': _Action(
                'Search saved Media Library results',
                'Find saved images, videos and audio. Filter by date, rating, model, LoRA, tag or saved folder. '
                'Use library.folders.list to discover folder IDs. '
                'Read image settings with generation.history.get and completed image bytes with generation.images.get. '
                'Read video or audio settings and playback references with media.history.get '
                '(kind=video for video, or kind=audio for audio).',
                LibrarySearch, LibrarySearchResult, True, self._search_library),
            'library.folders.list': _Action(
                'List saved Media Library folders',
                'Read the existing folders and their item counts. Use folder:<id> to filter library.search. '
                'Do not guess a folder ID from a displayed name.',
                DraftList, GenerationFolderResponse, True, self._list_library_folders),
            'library.items.update': _Action(
                'Update a saved Library item',
                'Set the requested rating, favorite, tags, note or folder of the exact saved generation_id. '
                'Use library.search to resolve the target and confirm any ambiguous replacement with the user. '
                'Unspecified fields and the file itself stay unchanged. Use the same request_id only for a retry.',
                LibraryCurationUpdate, LibraryCurationResult, False, self._update_library_item),
            'library.items.update_many': _Action(
                'Organize selected Library items',
                'Set the rating, favorite, added or removed tags, or folder of the exact selected generation_ids. '
                'Use library.search to resolve the targets and ask the user to choose when ambiguous. '
                'If a selected item is missing, none are changed. Notes and media files are not replaced. '
                'Use the same request_id only when retrying this request.',
                LibraryBulkCurationUpdate, LibraryBulkCurationResult, False, self._update_library_items),
            'library.items.compare': _Action(
                'Compare saved conditions',
                'Compare selected Library IDs in order. List the actual saved prompt, model, LoRA and condition differences '
                'without starting generation. Recorded and requested conditions are separate. Use existing media readers '
                'for the actual files; this does not judge the appearance of the results.',
                LibraryComparisonGet, LibraryComparisonResult, True, self._compare_library_items),
            'library.parts.compare': _Action(
                'Read saved body-part review candidates',
                'Read existing Library candidates for the exact saved image artifacts, in the given order. '
                'Use generation.history.get to find the artifact ID. Saved corrections and exclusions apply. '
                'A null review means no current saved review, not that a body part is absent. '
                'These are suggestions, not judgments of anatomy or person ownership. This does not run detection.',
                LibraryPartsCompare, LibraryPartsResult, True, parts.compare),
            'library.parts.crop': _Action(
                'View one reviewed body-part candidate',
                'Use the image, review and candidate IDs returned by library.parts.compare to view one saved rectangle. '
                'Returns an image crop, not a mask or an assessment. This does not change the source or run detection.',
                LibraryPartCrop, LibraryPartCropResult, True, parts.crop),
            'library.parts.detect': _Action(
                'Find body-part candidates in a Library image',
                'Run the optional existing detector only for an explicitly selected image. '
                'Use generation.history.get to find the artifact ID. It does not install models or judge anatomy. '
                'Reuse request_id only when resending the same operation.',
                LibraryPartDetect, LibraryPartDetectionResult, False, parts.detect),
            'library.parts.mask.prepare': _Action(
                'Prepare an edit mask for chosen body-part candidates',
                'Use the exact image, review and chosen candidate IDs; do not guess the selection. '
                'Uses SAM if needed. Use the returned init_image_path and mask.path in a named inpaint draft. '
                'No existing draft is overwritten and no image generation starts. '
                'Reuse request_id only when resending the same operation.',
                LibraryPartMaskPrepare, LibraryPartMaskResult, False, parts.prepare_mask),
            'library.items.export': _Action(
                'Export a saved Library item as a ZIP',
                'Export the exact generation_id and saved settings into the existing local Library export directory. '
                'No external transfer and no replacement of earlier exports. Report missing files. '
                'Use download_path only in Asteria; external clients may not be able to open it. '
                'Reuse request_id only when resending this exact request.',
                GenerationHistoryGet, LibraryExportResult, False, self._export_library_item),
            'library.items.delete': _Action(
                'Delete selected Library history',
                'Delete only the exact Library IDs the user asked to remove. If ambiguous, use library.search and ask which items. '
                'By default keep files; set delete_files only when requested. The same request_id never repeats removal. '
                'Report protected or unconfirmed files separately from deletion of history.',
                LibraryDelete, LibraryDeleteResult, False, self._delete_library_items),
            'generation.history.get': _Action(
                'Read a saved image generation',
                'Read the actual settings, input paths and output identities from the existing Asteria history. '
                'This does not depend on the external conversation and does not replace any screen input.',
                GenerationHistoryGet, GenerationHistoryItem, True, self._history),
            'generation.history.restore_draft': _Action(
                'Save a new draft from image generation history',
                'Restore saved settings and original image or mask inputs from the selected history. '
                'To reuse the Seed of a particular image, use its saved primary output_index. '
                'The original stays unchanged. This saves only; use generation.start with the returned revision '
                'only if a new generation was requested.',
                RestoreGenerationDraft, AssistantDraftRecord, False, self._restore_history_draft),
            'generation.images.get': _Action(
                'Read a result image',
                'Read actual image bytes using a generation_id and index in its output_paths. '
                'The returned bytes and hash belong to the same file shown in Asteria Library.',
                GenerationImageGet, GenerationImageContent, True, self._result_image),
        }
        self._actions.update(MediaActions(config, self._repository).catalog())
        self._actions.update(SequenceActions(config).catalog())
        self._actions.update(LibraryDelivery(config).catalog())

    def catalog(self) -> list[ActionDescription]:
        return [ActionDescription(
            name=name, title=action.title, description=action.description,
            read_only=action.read_only, input_schema=action.input_model.model_json_schema(),
            output_schema=action.output_model.model_json_schema(),
        ) for name, action in self._actions.items()]

    def invoke(self, name: str, call: ActionCall) -> ActionResult:
        action = self._actions.get(name)
        if action is None:
            raise ActionError('action_not_found', 404, action=name)
        if not action.read_only and not call.request_id:
            raise ActionError('action_request_id_required', 422)
        arguments = action.input_model.model_validate(call.arguments)
        try:
            return action.run(arguments, call.request_id)
        except DraftNotFound as exc:
            raise ActionError('assistant_draft_not_found', 404, draft_id=str(exc)) from exc
        except DraftRevisionNotFound as exc:
            raise ActionError('assistant_draft_revision_not_found', 404,
                              draft_id=exc.draft_id, revision=exc.revision) from exc
        except DraftConflict as exc:
            detail = {'current': exc.current.action_data()} if exc.current else {}
            raise ActionError(exc.code, 409, **detail) from exc
        except ActionRequestConflict as exc:
            raise ActionError('action_request_conflict', 409) from exc
        except OperationConflictError as exc:
            raise ActionError(str(exc), 409) from exc

    def _repository(self) -> AssistantDraftRepository:
        return AssistantDraftRepository(self.config.data_dir / 'asteria.db')

    @staticmethod
    def _image_draft(draft: AssistantDraftRecord):
        if draft.document.get('mode') not in {'txt2img', 'img2img', 'inpaint'}:
            raise ActionError('assistant_draft_kind_mismatch', 422)
        return TypeAdapter(GenerationDraftDocument).validate_python(draft.document)

    def _image(self, request: ImageGet, _request_id: str | None) -> ActionResult:
        from app.services.action_images import ActionImageRepository
        record = ActionImageRepository(self.config.data_dir).get(request.asset_id)
        return ActionResult(data=record.model_dump(mode='json'))

    def _search_library(self, request: LibrarySearch, _request_id: str | None) -> ActionResult:
        from app.services.history import HistoryRepository

        try:
            page = HistoryRepository.page_existing(
                self.config.data_dir / 'asteria.db',
                query=request.q, media_kind=request.media_kind, profile=request.profile,
                tag=request.tag, favorite=request.favorite, folder=request.folder,
                limit=request.limit, cursor=request.cursor,
                date_from=request.date_from, date_to=request.date_to, utc_offset_minutes=request.utc_offset_minutes,
                rating_min=request.rating_min, rating_max=request.rating_max, model=request.model, lora=request.lora,
                condition_verdict=request.condition_verdict,
            )
        except ValueError as exc:
            if str(exc) in {'invalid_history_cursor', 'invalid_history_filter_range'}:
                raise ActionError(str(exc), 400) from exc
            raise
        result = LibrarySearchResult(
            items=[LibrarySearchItem.model_validate(dict(row, generation_id=row['id'])) for row in page.items],
            next_cursor=page.next_cursor,
            has_more=page.has_more,
        )
        return ActionResult(data=result.model_dump(mode='json'))

    def _list_library_folders(self, _request: DraftList, _request_id: str | None) -> ActionResult:
        from app.schemas.generation import GenerationFolderResponse
        from app.services.history import HistoryRepository

        rows = HistoryRepository.list_folders_existing(self.config.data_dir / 'asteria.db')
        return ActionResult(data=GenerationFolderResponse(items=rows).model_dump(mode='json'))

    def _export_library_item(self, request: GenerationHistoryGet, request_id: str) -> ActionResult:
        from app.services.library_exports import export_library_item
        return export_library_item(self.config.data_dir, request, request_id)

    def _update_library_item(self, request: LibraryCurationUpdate, request_id: str) -> ActionResult:
        from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
        from app.services.history import HistoryRepository
        from app.services.library_curation import update_library_curation
        from app.services.operation_repository import canonical_operation_json

        action = 'library.items.update'
        arguments = canonical_operation_json(request.model_dump(mode='json', exclude_unset=True))
        repository = HistoryRepository(self.config.data_dir / 'asteria.db')
        with repository.publication() as connection:
            create_action_receipt_table(connection)
            receipt = read_action_receipt(connection, request_id, action, arguments)
            if receipt is not None:
                return ActionResult(data=receipt, replayed=True)
            try:
                item = update_library_curation(repository, request.generation_id, request,
                                               update_folder='folder_id' in request.model_fields_set)
            except ValueError as exc:
                if str(exc) == 'folder_not_found':
                    raise ActionError('folder_not_found', 404) from exc
                raise
            if item is None:
                raise ActionError('generation_not_found', 404)
            result = LibraryCurationResult.model_validate(dict(item, generation_id=item['id']))
            save_action_receipt(connection, request_id, action, arguments, result.model_dump_json())
        return ActionResult(data=result.model_dump(mode='json'))

    def _update_library_items(self, request: LibraryBulkCurationUpdate, request_id: str) -> ActionResult:
        from app.services import api_errors
        from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
        from app.services.history import (
            MAX_USER_TAGS, HistoryCurationTargetsMissingError, HistoryRepository, HistoryTagLimitExceededError,
        )
        from app.services.library_curation import sync_library_curation_sidecars
        from app.services.operation_repository import canonical_operation_json

        name = 'library.items.update_many'
        arguments = canonical_operation_json(request.model_dump(mode='json', exclude_unset=True))
        history = HistoryRepository(self.config.data_dir / 'asteria.db')
        with history.publication() as connection:
            create_action_receipt_table(connection)
            receipt = read_action_receipt(connection, request_id, name, arguments)
            if receipt is not None:
                return ActionResult(data=receipt, replayed=True)
            try:
                items = history.bulk_update_curation(
                    request.generation_ids, rating=request.rating, favorite=request.favorite,
                    tags_add=request.tags_add, tags_remove=request.tags_remove,
                    folder_id=request.folder_id, update_folder='folder_id' in request.model_fields_set, require_all=True,
                )
            except HistoryCurationTargetsMissingError as exc:
                raise ActionError(api_errors.GENERATION_NOT_FOUND, 404, missing_ids=exc.missing_ids) from exc
            except HistoryTagLimitExceededError as exc:
                raise ActionError(api_errors.HISTORY_TAG_LIMIT_EXCEEDED, 400, limit=MAX_USER_TAGS) from exc
            except ValueError as exc:
                if str(exc) == 'folder_not_found':
                    raise ActionError('folder_not_found', 404) from exc
                raise
            sync_library_curation_sidecars(items)
            result = LibraryBulkCurationResult(items=[
                LibraryCurationResult.model_validate(dict(item, generation_id=item['id'])) for item in items
            ])
            save_action_receipt(connection, request_id, name, arguments, result.model_dump_json())
        return ActionResult(data=result.model_dump(mode='json'))

    def _compare_library_items(self, request: LibraryComparisonGet, _request_id: str | None) -> ActionResult:
        from app.services.api_errors import GENERATION_NOT_FOUND
        from app.services.history import HistoryRepository
        from app.services.library_comparison import compare_library_items

        rows = HistoryRepository.get_many_existing(self.config.data_dir / 'asteria.db', request.generation_ids)
        if len(rows) != len(request.generation_ids):
            found = {row['id'] for row in rows}
            raise ActionError(GENERATION_NOT_FOUND, 404, missing_ids=[value for value in request.generation_ids if value not in found])
        return ActionResult(data=compare_library_items(rows).model_dump(mode='json'))

    def _delete_library_items(self, request: LibraryDelete, request_id: str) -> ActionResult:
        from app.services.library_deletion import delete_library_items

        return delete_library_items(self.config.data_dir, request, request_id)

    def _history(self, request: GenerationHistoryGet, _request_id: str | None) -> ActionResult:
        from fastapi import HTTPException
        from app.schemas.generation import GenerationHistoryItem
        from app.services.history import HistoryRepository

        record = HistoryRepository(self.config.data_dir / 'asteria.db').get(request.generation_id)
        if record is None:
            raise HTTPException(404, detail={'code': 'generation_not_found'})
        return ActionResult(data=GenerationHistoryItem.model_validate(record).model_dump(mode='json'))

    def _restore_history_draft(self, request: RestoreGenerationDraft, request_id: str) -> ActionResult:
        import sqlite3

        from app.services.generation_history_restore import HistoryRestoreError, restore_image_generation
        from app.services.history import HistoryRepository

        db = self.config.data_dir / 'asteria.db'
        operations = OperationRepository(db)

        def document() -> dict:
            try:
                item = HistoryRepository.registered_output_source(db, request.generation_id, include_model_settings=True)
            except (sqlite3.Error, OSError, ValueError) as exc:
                raise HistoryRestoreError('generation_history_input_unavailable') from exc
            return restore_image_generation(item, operations, output_index=request.output_index)

        try:
            draft, replayed = self._repository().restore_history(request_id, request.generation_id, request.title,
                document, output_index=request.output_index)
        except HistoryRestoreError as exc:
            raise ActionError(str(exc), exc.status_code, **exc.detail) from exc
        return ActionResult(data=draft.model_dump(mode='json'), replayed=replayed)

    def _result_image(self, request: GenerationImageGet, _request_id: str | None) -> ActionResult:
        import base64
        from hashlib import sha256
        from io import BytesIO
        from pathlib import Path
        from fastapi import HTTPException
        from PIL import Image
        from app.schemas.action_contracts import InlineImage
        from app.schemas.media import image_media_type

        generation = self._generation_runtime()
        record = self._history(request, None).data
        if record['media_kind'] != 'image' or request.image_index >= len(record['output_paths']):
            raise HTTPException(404, detail={'code': 'output_image_not_found'})
        response = generation.output_file_by_path(record['output_paths'][request.image_index])
        path = Path(response.path)
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise HTTPException(404, detail={'code': 'output_image_not_found'})
        try:
            data = path.read_bytes()
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
        except (OSError, ValueError):
            raise HTTPException(404, detail={'code': 'output_image_not_found'}) from None
        content = GenerationImageContent(generation_id=request.generation_id, image_index=request.image_index,
            path=str(path), mime_type=image_media_type(str(path)), sha256=sha256(data).hexdigest(), width=width, height=height,
            image=InlineImage(mime_type=image_media_type(str(path)), data_base64=base64.b64encode(data).decode('ascii')))
        return ActionResult(data=content.model_dump(mode='json'))

    def _profiles(self, _request: DraftList, _request_id: str | None) -> ActionResult:
        from app.api.models import ModelProfileListResponse, profile_to_response
        from app.services.model_profiles import ModelProfileRepository

        result = ModelProfileListResponse(profiles=[profile_to_response(profile) for profile in
            ModelProfileRepository(self.config.model_profiles_path).list()])
        return ActionResult(data=result.model_dump(mode='json'))

    def _library(self, request: ModelLibraryQuery, _request_id: str | None) -> ActionResult:
        from app.api.models import ModelLibraryResponse, library_item_to_response, link_health_to_response
        from app.services.model_library_cache import get_model_library
        from app.services.model_storage import resolve_model_root

        root = resolve_model_root(data_dir=self.config.data_dir, fallback=self.config.model_dir)
        library = get_model_library(root, config=self.config)
        items = library.by_type(request.type) if request.type else library.items
        result = ModelLibraryResponse(items=[library_item_to_response(item) for item in items],
                                      link_health=link_health_to_response(library.link_health))
        return ActionResult(data=result.model_dump(mode='json'))

    def _schema(self, request: GenerationSchemaQuery, _request_id: str | None) -> ActionResult:
        from app.api.generation import load_generation_schema

        return ActionResult(data=load_generation_schema(request.profile, self.config).model_dump(mode='json'))

    def _presets(self, request: GenerationPresetQuery, _request_id: str | None) -> ActionResult:
        from app.services.generation_preset_catalog import read_generation_presets

        return ActionResult(data=read_generation_presets(self.config, request).model_dump(mode='json', exclude_none=True))

    def _create(self, request: DraftCreate, request_id: str) -> ActionResult:
        record, replayed = self._repository().create(request_id, request.title, request.document.model_dump(mode='json'))
        return ActionResult(data=record.model_dump(mode='json'), replayed=replayed)

    def _update(self, request: DraftUpdate, request_id: str) -> ActionResult:
        record, replayed = self._repository().update(
            request_id, request.draft_id, request.expected_revision, request.title,
            request.document.model_dump(mode='json'),
        )
        return ActionResult(data=record.model_dump(mode='json'), replayed=replayed)

    def _get(self, request: DraftGet, _request_id: str | None) -> ActionResult:
        return ActionResult(data=self._repository().get(request.draft_id).action_data())

    def _validate_draft(self, request: DraftRevisionGet, _request_id: str | None) -> ActionResult:
        from app.services.action_draft_images import inspect_draft_inputs
        from app.services.model_library_cache import get_model_library
        from app.services.model_selections import ModelSelectionRepository

        generation = self._generation_runtime()
        draft = self._repository().get_revision(request.draft_id, request.revision)
        document = self._image_draft(draft)
        inputs, mode = document.input, document.mode
        generation._validate_generation_action_mode(mode, inputs)
        runtime = generation._selected_runtime_id(self.config, inputs.runtime)
        generation._preflight_profile_prompt_request(inputs.model_copy(deep=True))
        slots = generation._model_slots_for_request(inputs, self.config)
        selected = set(ModelSelectionRepository(self.config.model_selections_path).get(inputs.profile).slots.values())
        catalog_paths: set[str] | None = None

        def safe_model_file_exists(path: str) -> bool | None:
            nonlocal catalog_paths
            if path not in selected:
                if catalog_paths is None:
                    library = get_model_library(generation._model_root(self.config), config=self.config)
                    catalog_paths = {str(item.path) for item in library.items}
                if path not in catalog_paths:
                    return None
            return generation._path_is_file_safely(path)

        unverified_slots = generation._validate_required_model_slots(
            inputs.profile, slots, self.config, check_file=safe_model_file_exists)
        images, unverified_images = inspect_draft_inputs(self.config, draft) if mode != 'txt2img' else ([], [])
        matching = (images[0].width, images[0].height) == (images[1].width, images[1].height) if len(images) == 2 else None
        result = GenerationDraftValidation(
            draft_id=draft.id, revision=draft.revision, mode=mode, profile=inputs.profile, runtime=runtime,
            width=inputs.width, height=inputs.height, unverified_model_slots=unverified_slots,
            images=images, unverified_images=unverified_images, matching_image_dimensions=matching,
            not_checked=('model_loader_compatibility', 'optional_runtime_assets', 'runtime_load_and_execution'),
        )
        return ActionResult(data=result.model_dump(mode='json'))

    def _draft_image(self, request: DraftImageGet, _request_id: str | None) -> ActionResult:
        from app.services.action_draft_images import read_draft_image

        self._generation_runtime()
        draft = self._repository().get_revision(request.draft_id, request.revision)
        return ActionResult(data=read_draft_image(self.config, draft, request.role).model_dump(mode='json'))

    def _derive(self, request: DraftDerive, request_id: str) -> ActionResult:
        record, replayed = self._repository().derive(request_id, **request.model_dump())
        return ActionResult(data=record.action_data(), replayed=replayed)

    def _list(self, _request: DraftList, _request_id: str | None) -> ActionResult:
        return ActionResult(data={'items': [draft.action_data() for draft in self._repository().list()]})

    def _start(self, request: GenerationStart, request_id: str) -> ActionResult:
        generation = self._generation_runtime()

        arguments = request.model_dump(mode='json')
        operations = OperationRepository(self.config.data_dir / 'asteria.db')
        accepted = operations.get_action_operation(request_id, 'generation.start', arguments)
        if accepted is not None:
            return ActionResult(data=generation.get_job(accepted.operation_id).model_dump(mode='json'), replayed=True)
        draft = self._repository().get(request.draft_id)
        if draft.revision != request.expected_revision:
            raise DraftConflict('assistant_draft_revision_conflict', draft)
        document = self._image_draft(draft)
        # Capture existing model choices without changing global Generate settings.
        selected = generation._model_slots_for_request(document.input, self.config)
        adopted = document.input.model_copy(update={'model_slot_overrides': {
            **selected, **document.input.model_slot_overrides,
        }})
        result, replayed = generation.enqueue_action_generation(
            document.mode, adopted, request_id=request_id, arguments=arguments,
            adopted_draft={'id': draft.id, 'revision': draft.revision},
        )
        return ActionResult(data=result.model_dump(mode='json'), replayed=replayed)

    def _job(self, request: GenerationJobGet, _request_id: str | None) -> ActionResult:
        generation = self._generation_runtime()
        self._image_operation(request.job_id)
        return ActionResult(data=generation.get_job(request.job_id).model_dump(mode='json'))

    def _list_jobs(self, request: JobsList, _request_id: str | None) -> ActionResult:
        from app.services.action_video_sequence import SequenceActions
        from app.services.music_jobs import MUSIC_OPERATION_KIND
        from app.services.video_executor import VIDEO_OPERATION_KIND
        from app.services.video_sequence_jobs import VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND, VIDEO_SEQUENCE_CLIP_OPERATION_KIND
        from app.services.video_upscale_jobs import VIDEO_UPSCALE_OPERATION_KIND

        detail_actions: dict[str, tuple[JobMedia, str]] = {
            **{kind: ('image', 'generation.jobs.get') for kind in _IMAGE_JOB_KINDS},
            'generation.xyz': ('image', 'generation.xyz.get'),
            'generation.prompt_comparison': ('image', 'generation.comparisons.get'),
            VIDEO_OPERATION_KIND: ('video', 'media.jobs.get'),
            VIDEO_UPSCALE_OPERATION_KIND: ('video', 'media.jobs.get'),
            VIDEO_SEQUENCE_CLIP_OPERATION_KIND: ('video', 'video.sequences.jobs.get'),
            VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND: ('video', 'video.sequences.jobs.get'),
            MUSIC_OPERATION_KIND: ('music', 'media.jobs.get'),
        }
        kinds = [kind for kind, (media, _) in detail_actions.items() if request.media in (None, media)]
        operations = OperationRepository(self.config.data_dir / 'asteria.db').list(
            kinds=kinds, statuses=(request.status,) if request.status else None, limit=request.limit, for_display=True,
        )
        items = []
        for operation in operations:
            media, action = detail_actions[operation.kind]
            lookup: dict[str, str] = {'job_id': operation.operation_id}
            detail_action: str | None = action
            if action == 'generation.comparisons.get':
                comparison_id = operation.request_snapshot.get('comparison_id')
                lookup = {'comparison_id': comparison_id} if isinstance(comparison_id, str) and comparison_id else {}
            elif action == 'video.sequences.jobs.get':
                sequence_id = SequenceActions.job_sequence_id(operation)
                lookup = {**lookup, 'sequence_id': sequence_id} if sequence_id else {}
            if not lookup:
                detail_action = None
            items.append(ListedJob(
                job_id=operation.operation_id, kind=operation.kind, media=media, status=operation.status,
                phase=operation.phase, progress_current=operation.progress_current, progress_total=operation.progress_total,
                progress_unit=operation.progress_unit, artifact_refs=operation.artifact_refs,
                error_code=operation.error_code, warning_codes=operation.warning_codes, retryable=operation.retryable,
                detail_action=detail_action, detail_arguments=lookup,
            ))
        return ActionResult(data=JobsListResult(items=items).model_dump(mode='json'))

    @staticmethod
    def _comparison_result(plan: dict, replayed: bool = False) -> ActionResult:
        return ActionResult(data=PromptComparisonStatus.model_validate(plan).model_dump(mode='json'), replayed=replayed)

    def _prepare_comparison(self, request: PrepareGenerationComparison, request_id: str) -> ActionResult:
        self._generation_runtime()
        from app.api import prompt_comparisons

        receipt = ActionRequestIdentity(request_id, 'generation.comparisons.prepare', request.model_dump(mode='json'))
        repo = prompt_comparisons.repository()
        comparison_id = repo.find_action_comparison(receipt)
        if comparison_id is not None:
            return self._comparison_result(prompt_comparisons.get_comparison(comparison_id), True)
        draft = self._repository().get(request.draft_id)
        if draft.revision != request.expected_revision:
            raise DraftConflict('assistant_draft_revision_conflict', draft)
        document = self._image_draft(draft)
        comparison = PromptComparisonRequest(mode=document.mode, base=document.input, **request.comparison.model_dump())
        plan = prompt_comparisons.prepare_document(comparison, self.config)
        comparison_id, replayed = repo.create_for_action(plan, receipt)
        return self._comparison_result(prompt_comparisons.get_comparison(comparison_id), replayed)

    def _start_comparison(self, request: GenerationComparisonGet, request_id: str) -> ActionResult:
        self._generation_runtime()
        from app.api.prompt_comparisons import start_comparison

        return self._comparison_result(*start_comparison(request.comparison_id, ActionRequestIdentity(
            request_id, 'generation.comparisons.start', request.model_dump(mode='json'))))

    def _get_comparison(self, request: GenerationComparisonGet, _request_id: str | None) -> ActionResult:
        self._generation_runtime()
        from app.api.prompt_comparisons import get_comparison

        return self._comparison_result(get_comparison(request.comparison_id))

    def _cancel_comparison(self, request: GenerationComparisonGet, request_id: str) -> ActionResult:
        self._generation_runtime()
        from app.api.prompt_comparisons import cancel_comparison

        return self._comparison_result(*cancel_comparison(request.comparison_id, ActionRequestIdentity(
            request_id, 'generation.comparisons.cancel', request.model_dump(mode='json'))))

    def _get_branch(self, request: GenerationHistoryGet, _request_id: str | None) -> ActionResult:
        details = self._generation_runtime().generation_branch_details(request.generation_id)
        return ActionResult(data=GenerationBranchDetails.model_validate(details).model_dump(mode='json'))

    def _start_branch(self, request: GenerationBranchStart, request_id: str) -> ActionResult:
        generation = self._generation_runtime()
        action = 'generation.branches.start'
        arguments = request.model_dump(mode='json', exclude_unset=True)
        operations = OperationRepository(self.config.data_dir / 'asteria.db')
        accepted = operations.get_action_operation(request_id, action, arguments)
        if accepted is not None:
            return ActionResult(data=generation.get_job(accepted.operation_id).model_dump(mode='json'), replayed=True)
        _, model = generation._prepare_generation_branch(request.generation_id, arguments.get('changes', {}))
        job, replayed = generation.enqueue_action_generation('txt2img', model, request_id=request_id,
            arguments=arguments, action_name=action)
        return ActionResult(data=job.model_dump(mode='json'), replayed=replayed)

    @staticmethod
    def _xyz_status(operation, replayed: bool = False) -> ActionResult:
        items = [int(ref.removeprefix('history:')) for ref in operation.artifact_refs if ref.startswith('history:')]
        plot = items.pop(0) if items and operation.status in {'completed', 'completed_with_warnings'} else None
        data = GenerationXYZStatus(operation_id=operation.operation_id, status=operation.status, phase=operation.phase,
            progress_current=operation.progress_current, progress_total=operation.progress_total, error_code=operation.error_code,
            warning_codes=operation.warning_codes, plot_history_id=plot, cell_history_ids=items)
        return ActionResult(data=data.model_dump(mode='json'), replayed=replayed)

    def _start_xyz(self, request: GenerationXYZStart, request_id: str) -> ActionResult:
        generation = self._generation_runtime()
        coordinator = generation._generation_operation_coordinator()
        action = 'generation.xyz.start'
        arguments = request.model_dump(mode='json')
        accepted = coordinator.repository.get_action_operation(request_id, action, arguments)
        if accepted is not None:
            return self._xyz_status(accepted, True)
        draft = self._repository().get(request.draft_id)
        if draft.revision != request.expected_revision:
            raise DraftConflict('assistant_draft_revision_conflict', draft)
        document = self._image_draft(draft)
        slots = generation._model_slots_for_request(document.input, self.config)
        base = document.input.model_copy(update={'model_slot_overrides': {**slots, **document.input.model_slot_overrides}})
        plot = XYZPlotRequest(mode=document.mode, base=base, axes=request.axes, contact_sheet=request.contact_sheet)
        cells = generation._prepare_xyz_cells(plot)
        snapshot = {'request': plot.model_dump(mode='json'), 'action_draft': {'id': draft.id, 'revision': draft.revision},
                    'generation_operation_snapshot': generation._generation_operation_snapshot(plot.base)}
        operation, created = coordinator.enqueue_with_action_receipt(request_id=request_id, action=action,
            arguments=arguments, kind='generation.xyz', source_actor='assistant', request_snapshot=snapshot,
            progress_total=len(cells), progress_unit='images')
        return self._xyz_status(operation, not created)

    def _xyz_operation(self, job_id: str):
        try:
            operation = self._generation_runtime()._generation_operation_coordinator().get(job_id)
        except OperationNotFoundError as exc:
            raise ActionError('generation_job_not_found', 404) from exc
        if operation.kind != 'generation.xyz':
            raise ActionError('generation_job_not_found', 404)
        return operation

    def _get_xyz(self, request: GenerationJobGet, _request_id: str | None) -> ActionResult:
        return self._xyz_status(self._xyz_operation(request.job_id))

    def _cancel_xyz(self, request: GenerationJobGet, request_id: str) -> ActionResult:
        operation = self._xyz_operation(request.job_id)
        generation = self._generation_runtime()
        try:
            target, created = generation._generation_operation_coordinator().repository.bind_action_operation(
                request_id, 'generation.xyz.cancel', request.model_dump(mode='json'), operation.operation_id)
            return self._xyz_status(generation.cancel_generation_operation(target.operation_id), not created)
        except OperationNotFoundError as exc:
            raise ActionError('generation_job_not_found', 404) from exc

    def _repeat_start(self, request: RepeatStart, request_id: str) -> ActionResult:
        self._generation_runtime()
        from app.api.generation_repeats import get_repeat_service, prepare_repeat_input
        service = get_repeat_service()
        arguments = request.model_dump(mode='json')
        accepted = service.coordinator.repository.get_action_operation(request_id, 'generation.repeats.start', arguments)
        if accepted is not None:
            return ActionResult(data=service.get(accepted.operation_id).model_dump(mode='json'), replayed=True)
        draft = self._repository().get(request.draft_id)
        if draft.revision != request.expected_revision:
            raise DraftConflict('assistant_draft_revision_conflict', draft)
        document = prepare_repeat_input(self._image_draft(draft))
        try:
            record, replayed = service.start(request_id, document, request.limits, identity_arguments=arguments)
        except ValueError as exc:
            if str(exc) != 'generation_repeat_checker_required':
                raise
            raise ActionError(str(exc), 409) from exc
        return ActionResult(data=record.model_dump(mode='json'), replayed=replayed)

    def _repeat_get(self, request: RepeatGet, _request_id: str | None) -> ActionResult:
        self._generation_runtime()
        from app.api.generation_repeats import get_repeat
        return ActionResult(data=get_repeat(request.repeat_id).model_dump(mode='json'))

    def _repeat_update(self, request: RepeatUpdate, request_id: str) -> ActionResult:
        self._generation_runtime()
        from app.api.generation_repeats import update_repeat
        return update_repeat(request.repeat_id, RepeatInputUpdate(request_id=request_id,
            expected_revision=request.expected_revision, document=request.document))

    def _repeat_stop(self, request: RepeatStop, request_id: str) -> ActionResult:
        self._generation_runtime()
        from app.api.generation_repeats import stop_repeat_action
        return stop_repeat_action(request.repeat_id, RepeatStopRequest(request_id=request_id, stop_current=request.stop_current))

    def _image_operation(self, job_id: str):
        try:
            operation = OperationRepository(self.config.data_dir / 'asteria.db').get(job_id)
        except OperationNotFoundError as exc:
            raise ActionError('generation_job_not_found', 404) from exc
        if operation.kind not in _IMAGE_JOB_KINDS:
            raise ActionError('generation_job_not_found', 404)
        return operation

    def _cancel_job(self, request: GenerationJobGet, request_id: str) -> ActionResult:
        generation = self._generation_runtime()
        self._image_operation(request.job_id)
        operations = OperationRepository(self.config.data_dir / 'asteria.db')
        try:
            operation, created = operations.bind_action_operation(
                request_id, 'generation.jobs.cancel', request.model_dump(mode='json'), request.job_id)
        except OperationNotFoundError as exc:
            raise ActionError('generation_job_not_found', 404) from exc
        # Redelivery also reapplies the existing idempotent stop: the process may
        # have exited after saving the target but before notifying the runtime.
        result = generation.cancel_job(operation.operation_id)
        return ActionResult(data=result.model_dump(mode='json'), replayed=not created)

    def _generation_runtime(self):
        from app.api import generation

        # The existing image executor is process-scoped. Never silently enqueue
        # into another AppRoot when this dispatcher was bound to a different DB.
        if self.config.data_dir.resolve() != generation._data_dir().resolve():
            raise ActionError('action_runtime_configuration_mismatch', 503)
        return generation

    def _retry_job(self, request: GenerationJobGet, request_id: str) -> ActionResult:
        job, replayed = self._generation_runtime().retry_generation_job(request.job_id, action_request_id=request_id)
        return ActionResult(data=job.model_dump(mode='json'), replayed=replayed)
