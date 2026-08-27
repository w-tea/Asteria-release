from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import AppConfig
from app.schemas.image_metadata import (
    ImageMetadataLibraryPreviewRequest,
    ImageMetadataPickerCancelled,
    ImageMetadataPickerResult,
    ImageMetadataPickerSelected,
    ImageMetadataPickerUnavailable,
    ImageMetadataPreview,
    ImageMetadataUseAsInputRequest,
    ImageMetadataUseAsInputResponse,
)
from app.services import api_errors
from app.services.advanced_styles import AdvancedStylesError, AdvancedStylesRepository
from app.services.history import HistoryRepository
from app.services.image_metadata_import import build_image_metadata_preview
from app.services.image_metadata_receipts import ImageMetadataReceipt, ImageMetadataReceiptError, ImageMetadataReceiptStore
from app.services.model_library_cache import get_model_library
from app.services.model_storage import resolve_model_root
from app.services.native_path_picker import pick_native_path


router = APIRouter(prefix="/api/image-metadata", tags=["image-metadata"])


def _receipt_store() -> ImageMetadataReceiptStore:
    return ImageMetadataReceiptStore(AppConfig().data_dir)


def _resource_context() -> tuple[object, set[str]]:
    config = AppConfig()
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    library = get_model_library(model_root)
    try:
        style_ids = {style.id for style in AdvancedStylesRepository(config.data_dir).list().styles}
    except AdvancedStylesError:
        style_ids = set()
    return library, style_ids


def _preview_from_receipt(
    receipt: ImageMetadataReceipt,
    *,
    history_payload: dict[str, object] | None = None,
    sidecar_path: Path | None = None,
) -> ImageMetadataPreview:
    library, style_ids = _resource_context()
    return build_image_metadata_preview(
        receipt.staged_path,
        preview_token=receipt.token,
        expires_at=receipt.expires_at,
        history_payload=history_payload,
        model_library=library,
        available_style_ids=style_ids,
        sidecar_path=sidecar_path,
    )


def _raise_receipt_error(exc: ImageMetadataReceiptError) -> None:
    status_code = 409
    if exc.code in {
        api_errors.IMAGE_METADATA_IMAGE_UNSUPPORTED,
        api_errors.IMAGE_METADATA_IMAGE_INVALID,
    }:
        status_code = 400
    elif exc.code in {
        api_errors.IMAGE_METADATA_IMAGE_UNAVAILABLE,
        api_errors.IMAGE_METADATA_RECEIPT_NOT_FOUND,
    }:
        status_code = 404
    raise HTTPException(status_code=status_code, detail={"code": exc.code}) from exc


@router.post("/pick-and-preview", response_model=ImageMetadataPickerResult)
def pick_and_preview_image_metadata() -> ImageMetadataPickerResult:
    status, selected = pick_native_path(
        target="file",
        title="Choose an image to reuse",
        initial_path=None,
        extensions=["png", "jpg", "jpeg", "webp"],
    )
    if status == "cancelled":
        return ImageMetadataPickerCancelled()
    if status != "selected" or not selected:
        return ImageMetadataPickerUnavailable()
    source = Path(selected)
    try:
        receipt = _receipt_store().stage_path(source)
        preview = _preview_from_receipt(receipt, sidecar_path=source.with_suffix(".asteria.json"))
    except ImageMetadataReceiptError as exc:
        _raise_receipt_error(exc)
    return ImageMetadataPickerSelected(preview=preview)


@router.post("/preview-upload", response_model=ImageMetadataPreview)
async def preview_uploaded_image_metadata(
    request: Request,
    filename: str = Query(min_length=1, max_length=260),
) -> ImageMetadataPreview:
    data = await request.body()
    try:
        receipt = _receipt_store().stage_bytes(data, filename=filename)
        return _preview_from_receipt(receipt)
    except ImageMetadataReceiptError as exc:
        _raise_receipt_error(exc)


@router.post("/preview-library", response_model=ImageMetadataPreview)
def preview_library_image_metadata(request: ImageMetadataLibraryPreviewRequest) -> ImageMetadataPreview:
    config = AppConfig()
    item = HistoryRepository(config.data_dir / "asteria.db").get(request.generation_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={"code": api_errors.IMAGE_METADATA_LIBRARY_OUTPUT_NOT_FOUND},
        )
    try:
        requested = Path(request.output_path).resolve(strict=True)
        registered = {
            Path(value).resolve(strict=True)
            for value in item.get("output_paths", [])
            if isinstance(value, str) and value.strip()
        }
    except OSError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": api_errors.IMAGE_METADATA_LIBRARY_OUTPUT_NOT_FOUND},
        ) from exc
    if requested not in registered or not requested.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": api_errors.IMAGE_METADATA_LIBRARY_OUTPUT_NOT_FOUND},
        )
    try:
        receipt = _receipt_store().stage_path(
            requested,
            generation_id=request.generation_id,
            output_path=str(requested),
        )
        return _preview_from_receipt(
            receipt,
            history_payload=item,
            sidecar_path=requested.with_suffix(".asteria.json"),
        )
    except ImageMetadataReceiptError as exc:
        _raise_receipt_error(exc)


@router.post("/use-as-input", response_model=ImageMetadataUseAsInputResponse)
def use_preview_image_as_input(request: ImageMetadataUseAsInputRequest) -> ImageMetadataUseAsInputResponse:
    try:
        asset = _receipt_store().consume_as_init_image(request.preview_token)
    except ImageMetadataReceiptError as exc:
        _raise_receipt_error(exc)
    return ImageMetadataUseAsInputResponse(path=str(asset.path), filename=asset.filename)
