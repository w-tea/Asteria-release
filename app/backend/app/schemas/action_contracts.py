"""Core action contracts shared by Assistant and MCP transports."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.generation import (
    MAX_LIBRARY_BULK_ITEMS, Img2ImgRequest, InpaintRequest, LibraryBulkCurationFields,
    LibraryCurationFields, Txt2ImgRequest, XYZAxis,
)
from app.schemas.media import with_exact_media_seed
from app.schemas.music import MusicRequest
from app.schemas.operations import OperationStatus
from app.schemas.prompt_comparison import PromptComparisonOptions
from app.schemas.provider_capabilities import ProviderCapabilityDescriptor
from app.schemas.resource_advisory import ResourceAdvisoryAuthorization
from app.schemas.video import VideoRequest
from app.schemas.video_upscale import VideoUpscaleRequest


class ActionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ActionCall(ActionInput):
    request_id: str | None = Field(default=None, min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ActionDescription(BaseModel):
    name: str
    title: str
    description: str
    read_only: bool
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class ActionResult(BaseModel):
    data: dict[str, Any]
    replayed: bool = False


class ImageDraft(ActionInput):
    @field_validator('input', mode='after', check_fields=False)
    @classmethod
    def use_generate_runtime_by_default(cls, value):
        # Generate explicitly selects this runtime. An action with no selection
        # must generate too, not fall through to the legacy packet-only adapter.
        return value.model_copy(update={'runtime': 'comfy_workflow'}) if value.runtime is None else value


class TextImageDraft(ImageDraft):
    kind: Literal['image_generation'] = 'image_generation'
    mode: Literal['txt2img']
    input: Txt2ImgRequest


class ImageImageDraft(ImageDraft):
    kind: Literal['image_generation'] = 'image_generation'
    mode: Literal['img2img']
    input: Img2ImgRequest


class InpaintDraft(ImageDraft):
    kind: Literal['image_generation'] = 'image_generation'
    mode: Literal['inpaint']
    input: InpaintRequest


GenerationDraftDocument = Annotated[
    TextImageDraft | ImageImageDraft | InpaintDraft, Field(discriminator='mode')
]


class VideoGenerationDraft(ActionInput):
    kind: Literal['video_generation'] = 'video_generation'
    mode: Literal['video']
    input: VideoRequest


class MusicGenerationDraft(ActionInput):
    kind: Literal['music_generation'] = 'music_generation'
    mode: Literal['music']
    input: MusicRequest


SavedGenerationDraftDocument = Annotated[
    GenerationDraftDocument | VideoGenerationDraft | MusicGenerationDraft, Field(discriminator='mode')
]


class DraftCreate(ActionInput):
    title: str
    document: GenerationDraftDocument


class DraftUpdate(DraftCreate):
    draft_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=1, strict=True)


class DraftDerive(ActionInput):
    source_draft_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=1, strict=True)
    title: str
    input_changes: dict[str, Any] = Field(default_factory=dict, description=(
        'Only input fields to change. All omitted fields retain their saved values. '
        'A nested object or array replaces that whole field; it is not recursively merged. '
        'Use generation.schema.get for the existing profile fields.'))


class DraftGet(ActionInput):
    draft_id: str = Field(min_length=1)


class DraftRevisionGet(DraftGet):
    revision: int = Field(ge=1, strict=True)


class DraftImageGet(DraftRevisionGet):
    role: Literal['init', 'mask']


class DraftImageDiagnostics(BaseModel):
    role: Literal['init', 'mask']
    width: int
    height: int
    mime_type: str


class GenerationDraftValidation(BaseModel):
    draft_id: str
    revision: int
    mode: Literal['txt2img', 'img2img', 'inpaint']
    profile: str
    runtime: str
    width: int
    height: int
    unverified_model_slots: tuple[str, ...]
    images: tuple[DraftImageDiagnostics, ...]
    unverified_images: tuple[Literal['init', 'mask'], ...]
    matching_image_dimensions: bool | None
    not_checked: tuple[str, ...]


class DraftList(ActionInput):
    pass


class ModelLibraryQuery(ActionInput):
    type: str | None = None


class GenerationSchemaQuery(ActionInput):
    profile: str


class GenerationStart(DraftGet):
    expected_revision: int = Field(ge=1, strict=True)


class SaveMediaDraft(ActionInput):
    title: str
    document: Annotated[VideoGenerationDraft | MusicGenerationDraft, Field(discriminator='mode')]
    existing: GenerationStart | None = Field(default=None, description=(
        'Omit to create a new draft. To update, specify the saved draft_id and its current expected_revision.'))


class PrepareGenerationComparison(GenerationStart):
    comparison: PromptComparisonOptions


class GenerationComparisonGet(ActionInput):
    comparison_id: str = Field(min_length=1)


class GenerationXYZStart(GenerationStart):
    axes: list[XYZAxis] = Field(min_length=1, max_length=3)
    contact_sheet: bool = True


class GenerationXYZStatus(BaseModel):
    operation_id: str
    status: OperationStatus
    phase: str
    progress_current: int
    progress_total: int | None
    error_code: str | None
    warning_codes: tuple[str, ...]
    plot_history_id: int | None
    cell_history_ids: list[int]


class GenerationJobGet(ActionInput):
    job_id: str = Field(min_length=1)


JobMedia = Literal['image', 'video', 'music']


class JobsList(ActionInput):
    media: JobMedia | None = None
    status: OperationStatus | None = None
    limit: int = Field(default=30, ge=1, le=1000)


class ListedJob(BaseModel):
    job_id: str
    kind: str
    media: JobMedia
    status: OperationStatus
    phase: str
    progress_current: int
    progress_total: int | None
    progress_unit: str
    artifact_refs: tuple[str, ...]
    error_code: str | None
    warning_codes: tuple[str, ...]
    retryable: bool
    detail_action: str | None
    detail_arguments: dict[str, str]


class JobsListResult(BaseModel):
    items: list[ListedJob]


MediaGenerationKind = Literal['video', 'audio']


class MediaResourceChoice(ActionInput):
    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode='after')
    def validate_authorization_choice(self):
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError('media resource authorization choices are mutually exclusive')
        return self


class MediaGenerationStart(GenerationStart, MediaResourceChoice):
    kind: MediaGenerationKind


class MediaGenerationRetry(GenerationJobGet, MediaResourceChoice):
    pass


class MediaCapabilityGet(ActionInput):
    kind: MediaGenerationKind


class MediaSetupStatus(BaseModel):
    feature: str
    state: str
    reason_codes: tuple[str, ...]


class MediaCapabilities(BaseModel):
    providers: tuple[ProviderCapabilityDescriptor, ...]
    setup: tuple[MediaSetupStatus, ...]


class MusicTimingSummary(BaseModel):
    requested_duration_seconds: float
    effective_duration_seconds: float
    effective_audio_frames: int
    provider_clamp_reason: str | None


class MusicDraftValidation(BaseModel):
    ready: bool
    reason_code: str | None
    timing: MusicTimingSummary


class MediaGenerationJob(BaseModel):
    operation_id: str
    kind: Literal['video.generate', 'music.generate', 'video.upscale']
    status: OperationStatus
    phase: str
    progress_current: int
    progress_total: int | None
    progress_unit: str
    error_code: str | None
    warning_codes: tuple[str, ...]
    retryable: bool
    retry_of: str | None
    generation_id: int
    generation_ids: tuple[int, ...]
    artifact_refs: tuple[str, ...]


class ImageUploadMetadata(ActionInput):
    filename: str
    role: Literal['init', 'mask', 'control'] = 'init'


class ImageGet(ActionInput):
    asset_id: str


class ActionImageRecord(BaseModel):
    asset_id: str
    path: str
    filename: str
    role: Literal['init', 'mask', 'control']
    mime_type: str
    sha256: str
    width: int
    height: int


class InlineImage(BaseModel):
    mime_type: str
    data_base64: str


class ActionImageContent(ActionImageRecord):
    image: InlineImage


class GenerationPresetQuery(ActionInput):
    kind: Literal['generation', 'prompt', 'style', 'wildcard']
    id: str = Field(default='', description='Read the values for this exact saved ID. Without an ID, list names and IDs.')
    query: str = Field(default='', description='Filter names and IDs; leave blank to list all.')
    profile: str = Field(default='', description='Optional profile filter for generation presets.')
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=25, ge=1, le=100, strict=True)


class SavedGenerationPreset(BaseModel):
    id: str
    name: str
    data: dict[str, Any] | None = None


class SavedGenerationPresetList(BaseModel):
    items: list[SavedGenerationPreset]
    next_offset: int | None = None


class GenerationHistoryGet(ActionInput):
    generation_id: int = Field(ge=1, strict=True)


class LibrarySearch(ActionInput):
    q: str = ''
    media_kind: Literal['all', 'image', 'video', 'audio'] = 'all'
    profile: str = ''
    tag: str = ''
    favorite: bool = False
    folder: str = Field(default='all', description='Use all, unclassified, or folder:<saved folder ID>.')
    date_from: date | None = Field(default=None, description='First included calendar day (YYYY-MM-DD).')
    date_to: date | None = Field(default=None, description='Last included calendar day (YYYY-MM-DD).')
    utc_offset_minutes: int = Field(default=0, ge=-1439, le=1439,
                                    description='Offset for the requested calendar days, e.g. 540 for JST. Defaults to UTC.')
    rating_min: int | None = Field(default=None, ge=0, le=5, description='Lowest included saved rating; 0 is unrated.')
    rating_max: int | None = Field(default=None, ge=0, le=5, description='Highest included saved rating; 0 is unrated.')
    model: str = Field(default='', description='Match a name recorded in the saved model slots.')
    lora: str = Field(default='', description='Match a saved LoRA name, not prompt text.')
    condition_verdict: Literal['', 'match', 'mismatch', 'unknown', 'unchecked'] = ''
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str = Field(default='', description='Use next_cursor from a page with the same filters.')


class LibrarySearchItem(BaseModel):
    generation_id: int
    media_kind: Literal['image', 'video', 'audio']
    status: str
    profile: str
    prompt: str
    tags: list[str]
    generated_tags: list[str]
    rating: int
    favorite: bool
    created_at: str
    folder_id: int | None
    folder_name: str | None


class LibrarySearchResult(BaseModel):
    items: list[LibrarySearchItem]
    next_cursor: str | None
    has_more: bool


class LibraryCurationUpdate(GenerationHistoryGet, LibraryCurationFields):
    tags: list[str] | None = Field(default=None, description='Replace the saved tags. Omit to keep them, or pass [] to clear.')
    notes: str | None = Field(default=None, description='Replace the saved note. Omit to keep it, or pass an empty string to clear.')
    folder_id: int | None = Field(default=None, description='Move to this saved folder ID. Omit to keep it, or pass null to unclassify.')


class LibraryCurationResult(BaseModel):
    generation_id: int
    rating: int
    favorite: bool
    tags: list[str]
    notes: str
    folder_id: int | None
    folder_name: str | None


LibraryRecordId = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]


class LibraryBulkCurationUpdate(ActionInput, LibraryBulkCurationFields):
    generation_ids: list[LibraryRecordId] = Field(
        min_length=1, max_length=MAX_LIBRARY_BULK_ITEMS, description='The exact confirmed saved Library item IDs.')
    folder_id: int | None = Field(default=None, description='Move to this saved folder ID. Omit to keep it, or pass null to unclassify.')


class LibraryBulkCurationResult(BaseModel):
    items: list[LibraryCurationResult]


class LibraryComparisonGet(ActionInput):
    generation_ids: list[LibraryRecordId] = Field(min_length=2, description='Saved item IDs in the requested comparison order.')

    @field_validator('generation_ids')
    @classmethod
    def distinct_items(cls, values: list[int]) -> list[int]:
        if len(set(values)) != len(values):
            raise ValueError('Use distinct saved Library items for comparison.')
        return values


class LibraryComparisonArtifact(BaseModel):
    artifact_id: str
    kind: Literal['image', 'video', 'audio']
    mime_type: str


class LibraryComparisonItem(BaseModel):
    generation_id: int
    media_kind: Literal['image', 'video', 'audio']
    profile: str
    prompt: str
    expanded_prompt: str
    negative_prompt: str
    model_slots: dict[str, str]
    loras: list[dict]
    recorded_conditions: dict[str, str | int | float | bool | None]
    requested_conditions: dict[str, str | int | float | bool | None]
    artifacts: list[LibraryComparisonArtifact]


class LibraryComparisonResult(BaseModel):
    items: list[LibraryComparisonItem]
    different_fields: list[str]


class LibraryExportResult(BaseModel):
    generation_id: int
    export_id: str
    status: Literal['unconfirmed', 'completed']
    download_path: str | None = None
    file_count: int = 0
    missing_count: int = 0
    bundle_size_bytes: int = 0
    bundle_sha256: str = ''


class LibraryDelete(ActionInput):
    generation_ids: list[LibraryRecordId] = Field(
        min_length=1, max_length=MAX_LIBRARY_BULK_ITEMS, description='The exact selected Library history IDs.')
    delete_files: bool = Field(default=False, description='Set true only when the user requests removal of files as well as history.')


class LibraryDeleteResult(BaseModel):
    deleted_ids: list[int]
    files_status: Literal['not_requested', 'unconfirmed', 'finished']
    deleted_files: int = 0
    missing_files: int = 0
    skipped_files: int = 0
    protected_files: int = 0


class RestoreGenerationDraft(GenerationHistoryGet):
    title: str
    output_index: int | None = Field(default=None, ge=0, strict=True, description=(
        'Output index from generation.history.get settings.primary_outputs, not the position in output_paths. '
        'Omit to use the history default.'))


class MediaHistoryGet(GenerationHistoryGet):
    kind: MediaGenerationKind


class RestoreMediaDraft(MediaHistoryGet):
    title: str


class MediaArtifactReference(BaseModel):
    artifact_id: str
    kind: Literal['video', 'audio', 'poster']
    role: str
    mime_type: str
    url: str
    poster_artifact_id: str | None = None
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    sha256: str | None = None


class MediaHistoryResult(BaseModel):
    generation_id: int
    kind: MediaGenerationKind
    status: str
    seed_decimal: str | None
    request: VideoRequest | MusicRequest | VideoUpscaleRequest | None
    sequence_id: str | None = None
    artifacts: tuple[MediaArtifactReference, ...]

    def action_data(self) -> dict[str, Any]:
        payload = self.model_dump(mode='json')
        if isinstance(self.request, VideoUpscaleRequest):
            request = payload['request']
            request['sampling'] = with_exact_media_seed(request['sampling'])
        elif self.request is not None:
            payload['request'] = with_exact_media_seed(payload['request'])
        return payload


class MediaSourceTarget(ActionInput):
    generation_id: int = Field(ge=1, strict=True)
    artifact_id: str = Field(min_length=1)


class MediaSourceRecord(MediaSourceTarget):
    filename: str
    kind: Literal['image', 'video', 'audio']
    mime_type: str
    sha256: str | None
    url: str
    poster_artifact_id: str | None


class VideoUpscaleSourceGet(GenerationHistoryGet):
    artifact_id: str = Field(min_length=1)


class VideoUpscaleSourceSummary(BaseModel):
    schema_version: Literal['asteria.video-upscale.source.v1']
    generation_id: int
    artifact_id: str
    available: bool
    source_width: int
    source_height: int
    frame_count: int
    fps_numerator: int
    fps_denominator: int
    has_audio: bool
    default_target_width: int
    default_target_height: int
    supported_modes: tuple[Literal['h3_3d_model', 'model_free'], ...]
    cfg_supported: bool
    cfg_fixed_value: int
    setup_ready: bool


class GenerationBranchChanges(ActionInput):
    prompt: str | None = None
    negative_prompt: str | None = None
    cfg_scale: float | None = Field(default=None, allow_inf_nan=False)


class GenerationBranchStart(GenerationHistoryGet):
    changes: GenerationBranchChanges = Field(default_factory=GenerationBranchChanges)


class GenerationBranchDetails(BaseModel):
    generation_id: int
    cut_step: int
    bytes: int
    prompt: str
    negative_prompt: str
    cfg_scale: float
    profile: str
    steps: int
    seed: int
    sampler: str
    scheduler: str
    width: int
    height: int


class GenerationImageGet(GenerationHistoryGet):
    image_index: int = Field(default=0, ge=0, strict=True)


class GenerationImageContent(BaseModel):
    generation_id: int
    image_index: int
    path: str
    mime_type: str
    sha256: str
    width: int
    height: int
    image: InlineImage


class AssistantDraftRecord(BaseModel):
    id: str
    revision: int
    title: str
    document: dict[str, Any]
    created_at: str
    updated_at: str

    def action_data(self) -> dict[str, Any]:
        payload = self.model_dump(mode='json')
        document = payload['document']
        request = document.get('input')
        if document.get('mode') in {'music', 'video'} and isinstance(request, dict):
            document['input'] = with_exact_media_seed(request)
        return payload


class DraftImageContent(BaseModel):
    draft_id: str
    revision: int
    role: Literal['init', 'mask']
    path: str
    mime_type: str
    sha256: str
    width: int
    height: int
    image: InlineImage


class AssistantDraftList(BaseModel):
    items: list[AssistantDraftRecord]
