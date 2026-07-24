from io import BytesIO
from pathlib import Path
import re
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, model_validator

from app.core.config import AppConfig
from app.schemas.image_edit import (
    ImageEditCropOperation,
    ImageEditLineage,
    ImageEditPercentRect,
    ImageEditPixelRect,
)
from app.services import api_errors
from app.services.history import HistoryRepository
from app.services.image_validation import (
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    InvalidImageContentError,
    SUPPORTED_IMAGE_FORMATS,
    validate_image_bytes,
)
from app.services.image_edit_lineage import (
    image_edit_artifact,
    image_edit_manifest_path,
    load_image_edit_lineage,
    write_image_edit_lineage,
)
from app.services.edit_asset_lifecycle import AssetMeasure, EditAssetLifecycleService
from app.services.image_editor_drafts import (
    DraftConflictError,
    ImageEditorDraft,
    ImageEditorDraftCommitRequest,
    ImageEditorDraftRepository,
)
from app.services.user_settings import UserSettingsRepository, resolve_output_root

router = APIRouter(prefix="/api/assets", tags=["assets"])

MAX_UPLOAD_IMAGE_BYTES = DEFAULT_MAX_IMAGE_BYTES
MAX_UPLOAD_IMAGE_PIXELS = DEFAULT_MAX_IMAGE_PIXELS


class UploadedAssetResponse(BaseModel):
    path: str
    filename: str


class CropInitImageRequest(BaseModel):
    path: str
    pixel_rect: ImageEditPixelRect | None = None
    source_sha256: str = Field(default="", pattern=r"^$|^[0-9a-f]{64}$")
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    filename: str = "asteria_crop.png"

    @model_validator(mode="after")
    def validate_crop_coordinates(self):
        legacy = (self.x, self.y, self.width, self.height)
        if self.pixel_rect is None and any(value is None for value in legacy):
            raise ValueError("pixel_rect or complete percentage coordinates are required")
        return self


class CropInitImageResponse(UploadedAssetResponse):
    lineage_path: str
    lineage: ImageEditLineage


class AssetMeasureResponse(BaseModel):
    count: int = 0
    bytes: int = 0


class EditAssetLifecycleResponse(BaseModel):
    editing_temp: AssetMeasureResponse
    safe_reclaimable: AssetMeasureResponse
    thumbnail_cache: AssetMeasureResponse
    protected: AssetMeasureResponse
    reference_health: str


class EditAssetCleanupResponse(BaseModel):
    removed: AssetMeasureResponse
    skipped: AssetMeasureResponse
    failed: AssetMeasureResponse
    reference_health: str


class ImageEditorDraftMaskStageResponse(BaseModel):
    stage_token: str = Field(pattern=r"^[0-9a-f]{32}$")


def _measure_response(measure: AssetMeasure) -> AssetMeasureResponse:
    return AssetMeasureResponse(count=measure.count, bytes=measure.bytes)


def _safe_filename(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
    return name or "asset.png"


async def _upload_image_asset(request: Request, filename: str, directory_name: str, label: str) -> UploadedAssetResponse:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail={"code": api_errors.ASSET_IMAGE_UNSUPPORTED, "label": label})
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail={"code": api_errors.ASSET_IMAGE_EMPTY, "label": label})
    try:
        validate_image_bytes(
            data,
            suffix,
            max_bytes=MAX_UPLOAD_IMAGE_BYTES,
            max_pixels=MAX_UPLOAD_IMAGE_PIXELS,
        )
    except InvalidImageContentError as exc:
        raise HTTPException(status_code=400, detail={"code": api_errors.ASSET_IMAGE_INVALID, "label": label}) from exc
    output_dir = AppConfig().data_dir / directory_name
    filename = f"{uuid4().hex}_{_safe_filename(filename)}"
    path = output_dir / filename
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.ASSET_IMAGE_SAVE_FAILED, "label": label}) from exc
    return UploadedAssetResponse(path=str(path), filename=filename)


def _asset_data_dir() -> Path:
    return AppConfig().data_dir


def _is_registered_output_path(path: Path) -> bool:
    data_dir = _asset_data_dir()
    settings = UserSettingsRepository(data_dir / "user-settings.json").load().output
    try:
        resolved = path.resolve()
        output_root = resolve_output_root(data_dir, settings).resolve()
    except OSError:
        return False
    if resolved == output_root or output_root in resolved.parents:
        return True
    referenced = HistoryRepository(data_dir / "asteria.db").referenced_output_paths()
    for value in referenced:
        try:
            if Path(value).resolve() == resolved:
                return True
        except (TypeError, OSError):
            continue
    return False


def _is_asteria_init_image_path(path: Path) -> bool:
    try:
        root = (_asset_data_dir() / "init_images").resolve()
        resolved = path.resolve()
    except OSError:
        return False
    return resolved.parent == root


def _init_image_destination(filename: str) -> tuple[Path, str]:
    output_dir = _asset_data_dir() / "init_images"
    safe_name = _safe_filename(filename)
    if Path(safe_name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        safe_name = f"{Path(safe_name).stem or 'asset'}.png"
    final_name = f"{uuid4().hex}_{safe_name}"
    return output_dir / final_name, final_name


def _save_init_image_bytes(path: Path, data: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_bytes(data)
        temporary_path.replace(path)
    except OSError as exc:
        for cleanup_path in (temporary_path, path):
            try:
                cleanup_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(status_code=409, detail={"code": api_errors.ASSET_IMAGE_SAVE_FAILED, "label": "init"}) from exc


@router.post("/control-images", response_model=UploadedAssetResponse)
async def upload_control_image(request: Request, filename: str) -> UploadedAssetResponse:
    return await _upload_image_asset(request, filename, "control_images", "control")


@router.post("/init-images", response_model=UploadedAssetResponse)
async def upload_init_image(request: Request, filename: str) -> UploadedAssetResponse:
    return await _upload_image_asset(request, filename, "init_images", "init")


@router.post("/init-images/crop", response_model=CropInitImageResponse)
async def crop_init_image(request: CropInitImageRequest) -> CropInitImageResponse:
    try:
        source = Path(request.path).resolve()
    except OSError as exc:
        raise HTTPException(status_code=404, detail={"code": api_errors.ASSET_SOURCE_IMAGE_NOT_FOUND}) from exc
    try:
        source_available = source.exists() and source.is_file()
    except OSError:
        source_available = False
    if not source_available:
        raise HTTPException(status_code=404, detail={"code": api_errors.ASSET_SOURCE_IMAGE_NOT_FOUND})
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail={"code": api_errors.ASSET_IMAGE_UNSUPPORTED, "label": "source"})
    if not (_is_registered_output_path(source) or _is_asteria_init_image_path(source)):
        raise HTTPException(status_code=403, detail={"code": api_errors.ASSET_SOURCE_IMAGE_NOT_REGISTERED})

    try:
        source_bytes = source.read_bytes()
        with Image.open(BytesIO(source_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            source_width, source_height = image.size
            if request.pixel_rect:
                left = min(request.pixel_rect.x, source_width)
                top = min(request.pixel_rect.y, source_height)
                right = min(source_width, left + request.pixel_rect.width)
                bottom = min(source_height, top + request.pixel_rect.height)
            else:
                x = min(max(request.x or 0, 0), 95)
                y = min(max(request.y or 0, 0), 95)
                width = min(max(request.width or 1, 1), 100 - x)
                height = min(max(request.height or 1, 1), 100 - y)
                left = round((x / 100) * source_width)
                top = round((y / 100) * source_height)
                right = min(source_width, left + max(1, round((width / 100) * source_width)))
                bottom = min(source_height, top + max(1, round((height / 100) * source_height)))
            crop_width = right - left
            crop_height = bottom - top
            if crop_width < 8 or crop_height < 8:
                raise HTTPException(status_code=400, detail={"code": api_errors.ASSET_CROP_RECT_INVALID})
            cropped = image.crop((left, top, right, bottom))
            buffer = BytesIO()
            cropped.save(buffer, format="PNG")
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=400, detail={"code": api_errors.ASSET_SOURCE_IMAGE_CROP_FAILED}) from exc

    source_artifact = image_edit_artifact(source, source_bytes, source_width, source_height)
    if request.source_sha256 and request.source_sha256 != source_artifact.sha256:
        raise HTTPException(status_code=409, detail={"code": api_errors.ASSET_SOURCE_IMAGE_LINEAGE_MISMATCH})
    try:
        previous_lineage = load_image_edit_lineage(source)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.ASSET_SOURCE_IMAGE_LINEAGE_MISMATCH}) from exc
    if previous_lineage and previous_lineage.current != source_artifact:
        raise HTTPException(status_code=409, detail={"code": api_errors.ASSET_SOURCE_IMAGE_LINEAGE_MISMATCH})
    if previous_lineage and len(previous_lineage.operations) >= 64:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.ASSET_IMAGE_LINEAGE_LIMIT_EXCEEDED, "max_operations": 64},
        )

    output_bytes = buffer.getvalue()
    destination, _ = _init_image_destination(request.filename)
    output_path = destination.resolve()
    output_artifact = image_edit_artifact(output_path, output_bytes, crop_width, crop_height)
    operation = ImageEditCropOperation(
        source=source_artifact,
        output=output_artifact,
        rect_pixels=ImageEditPixelRect(x=left, y=top, width=crop_width, height=crop_height),
        rect_percent=ImageEditPercentRect(
            x=(left / source_width) * 100,
            y=(top / source_height) * 100,
            width=(crop_width / source_width) * 100,
            height=(crop_height / source_height) * 100,
        ),
    )
    lineage = ImageEditLineage(
        root=previous_lineage.root if previous_lineage else source_artifact,
        current=output_artifact,
        operations=[*(previous_lineage.operations if previous_lineage else []), operation],
    )
    _save_init_image_bytes(output_path, output_bytes)
    try:
        lineage_path = write_image_edit_lineage(output_path, lineage)
    except OSError as exc:
        for path in (output_path, image_edit_manifest_path(output_path)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(status_code=409, detail={"code": api_errors.ASSET_IMAGE_LINEAGE_SAVE_FAILED}) from exc
    return CropInitImageResponse(
        path=output_artifact.path,
        filename=output_artifact.filename,
        lineage_path=str(lineage_path),
        lineage=lineage,
    )


@router.post("/mask-images", response_model=UploadedAssetResponse)
async def upload_mask_image(request: Request, filename: str) -> UploadedAssetResponse:
    return await _upload_image_asset(request, filename, "mask_images", "mask")


@router.get("/lifecycle", response_model=EditAssetLifecycleResponse)
def edit_asset_lifecycle() -> EditAssetLifecycleResponse:
    inventory = EditAssetLifecycleService(_asset_data_dir()).inventory()
    return EditAssetLifecycleResponse(
        editing_temp=_measure_response(inventory.editing_temp),
        safe_reclaimable=_measure_response(inventory.safe_reclaimable),
        thumbnail_cache=_measure_response(inventory.thumbnail_cache),
        protected=_measure_response(inventory.protected),
        reference_health=inventory.reference_health,
    )


@router.post("/lifecycle/cleanup", response_model=EditAssetCleanupResponse)
def cleanup_edit_assets() -> EditAssetCleanupResponse:
    result = EditAssetLifecycleService(_asset_data_dir()).cleanup()
    if result is None:
        raise HTTPException(status_code=409, detail={"code": api_errors.ASSET_LIFECYCLE_BUSY})
    return EditAssetCleanupResponse(
        removed=_measure_response(result.removed),
        skipped=_measure_response(result.skipped),
        failed=_measure_response(result.failed),
        reference_health=result.reference_health,
    )


def _draft_repository() -> ImageEditorDraftRepository:
    return ImageEditorDraftRepository(_asset_data_dir())


def _draft_conflict(exc: DraftConflictError) -> HTTPException:
    return HTTPException(status_code=409, detail={"code": exc.code})


@router.get("/editor-draft", response_model=ImageEditorDraft, status_code=200)
def get_image_editor_draft():
    try:
        draft = _draft_repository().load()
    except DraftConflictError as exc:
        raise _draft_conflict(exc) from exc
    if draft is None:
        from fastapi import Response
        return Response(status_code=204)
    return draft


@router.post("/editor-draft/mask-stage", response_model=ImageEditorDraftMaskStageResponse)
async def stage_image_editor_draft_mask(request: Request) -> ImageEditorDraftMaskStageResponse:
    data = await request.body()
    try:
        validate_image_bytes(
            data,
            ".png",
            max_bytes=MAX_UPLOAD_IMAGE_BYTES,
            max_pixels=MAX_UPLOAD_IMAGE_PIXELS,
        )
    except InvalidImageContentError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.ASSET_IMAGE_INVALID, "label": "draft-mask"},
        ) from exc
    try:
        token = _draft_repository().stage_mask(data)
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED},
        ) from exc
    return ImageEditorDraftMaskStageResponse(stage_token=token)


@router.delete("/editor-draft/mask-stage/{stage_token}", status_code=204)
def cancel_image_editor_draft_mask_stage(stage_token: str) -> Response:
    try:
        _draft_repository().cancel_stage(stage_token)
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED},
        ) from exc
    return Response(status_code=204)


@router.post("/editor-draft/commit", response_model=ImageEditorDraft)
def commit_image_editor_draft(request: ImageEditorDraftCommitRequest) -> ImageEditorDraft:
    import time

    try:
        return _draft_repository().commit(request, updated_at=time.time())
    except DraftConflictError as exc:
        raise _draft_conflict(exc) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED},
        ) from exc


@router.post("/editor-draft/replace", response_model=ImageEditorDraft)
def replace_image_editor_draft() -> ImageEditorDraft:
    raise HTTPException(
        status_code=409,
        detail={"code": api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED},
    )


@router.put("/editor-draft", response_model=ImageEditorDraft)
def update_image_editor_draft() -> ImageEditorDraft:
    raise HTTPException(
        status_code=409,
        detail={"code": api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED},
    )


@router.post("/editor-draft/mask", response_model=ImageEditorDraft)
async def save_image_editor_draft_mask() -> ImageEditorDraft:
    raise HTTPException(
        status_code=409,
        detail={"code": api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED},
    )


@router.delete("/editor-draft/{draft_id}", status_code=204)
def discard_image_editor_draft(draft_id: str, revision: int):
    from fastapi import Response
    try:
        _draft_repository().discard(draft_id, revision)
    except DraftConflictError as exc:
        raise _draft_conflict(exc) from exc
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED}) from exc
    return Response(status_code=204)


@router.get("/{directory_name}/{filename}")
def asset_file(directory_name: str, filename: str) -> FileResponse:
    if directory_name not in {"control_images", "init_images", "mask_images", "editor_drafts"}:
        raise HTTPException(status_code=404, detail={"code": api_errors.ASSET_NOT_FOUND})
    if directory_name == "editor_drafts" and not re.fullmatch(r"[0-9a-f]{32}_[0-9]+_mask\.png", filename):
        raise HTTPException(status_code=404, detail={"code": api_errors.ASSET_NOT_FOUND})
    try:
        root = (AppConfig().data_dir / directory_name).resolve()
        path = (root / Path(filename).name).resolve()
    except OSError as exc:
        raise HTTPException(status_code=404, detail={"code": api_errors.ASSET_NOT_FOUND}) from exc
    try:
        available = path.exists() and path.is_file()
    except OSError:
        available = False
    if root != path.parent or not available:
        raise HTTPException(status_code=404, detail={"code": api_errors.ASSET_NOT_FOUND})
    return FileResponse(path, headers={"Access-Control-Allow-Origin": "*"})
