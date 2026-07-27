from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import AppConfig
from app.services.advanced_styles import AdvancedStyle, AdvancedStylesError, AdvancedStylesRepository

router = APIRouter(prefix="/api/advanced-styles", tags=["advanced-styles"])


class AdvancedStyleResponse(BaseModel):
    id: str
    group: str
    name: str
    prompt: str
    negative: str
    thumbnail_path: str | None = None


class AdvancedStylesResponse(BaseModel):
    groups: list[str]
    styles: list[AdvancedStyleResponse]
    source_path: Path


class AdvancedStyleSaveRequest(BaseModel):
    group: str
    name: str
    prompt: str = ""
    negative: str = ""


class AdvancedStyleCsvImportRequest(BaseModel):
    target_group: str = "default"
    csv_text: str
    update_existing: bool = True


class AdvancedStyleCsvImportResponse(BaseModel):
    imported: int


class AdvancedStyleThumbnailResponse(BaseModel):
    thumbnail_path: str | None = None
    deleted: bool = False


def get_repository() -> AdvancedStylesRepository:
    return AdvancedStylesRepository(AppConfig().data_dir)


def style_to_response(style: AdvancedStyle) -> AdvancedStyleResponse:
    return AdvancedStyleResponse(**style.__dict__)


@router.get("", response_model=AdvancedStylesResponse)
def list_styles() -> AdvancedStylesResponse:
    library = get_repository().list()
    return AdvancedStylesResponse(
        groups=library.groups,
        styles=[style_to_response(style) for style in library.styles],
        source_path=library.source_path,
    )


@router.post("/style", response_model=AdvancedStyleResponse)
def save_style(request: AdvancedStyleSaveRequest) -> AdvancedStyleResponse:
    if not request.group.strip() or not request.name.strip():
        raise HTTPException(status_code=400, detail="Style group and name are required")
    try:
        return style_to_response(
            get_repository().save_style(
                group=request.group,
                name=request.name,
                prompt=request.prompt,
                negative=request.negative,
            )
        )
    except AdvancedStylesError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "advanced_styles_persistence_failed"},
        ) from exc


@router.delete("/style", response_model=dict[str, bool])
def delete_style(group: str, name: str) -> dict[str, bool]:
    try:
        return {"deleted": get_repository().delete_style(group=group, name=name)}
    except AdvancedStylesError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "advanced_styles_persistence_failed"},
        ) from exc


@router.post("/style/thumbnail", response_model=AdvancedStyleThumbnailResponse)
async def upload_style_thumbnail(group: str, name: str, filename: str, body: bytes = Body()) -> AdvancedStyleThumbnailResponse:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported thumbnail image type")
    try:
        thumbnail_path = get_repository().save_thumbnail(group=group, name=name, image_bytes=body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AdvancedStylesError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "advanced_styles_persistence_failed"},
        ) from exc
    return AdvancedStyleThumbnailResponse(thumbnail_path=thumbnail_path)


@router.delete("/style/thumbnail", response_model=AdvancedStyleThumbnailResponse)
def delete_style_thumbnail(group: str, name: str) -> AdvancedStyleThumbnailResponse:
    try:
        return AdvancedStyleThumbnailResponse(deleted=get_repository().delete_thumbnail(group=group, name=name))
    except AdvancedStylesError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "advanced_styles_persistence_failed"},
        ) from exc


@router.post("/import-csv", response_model=AdvancedStyleCsvImportResponse)
def import_csv(request: AdvancedStyleCsvImportRequest) -> AdvancedStyleCsvImportResponse:
    try:
        imported = get_repository().import_csv(
            csv_text=request.csv_text,
            target_group=request.target_group,
            update_existing=request.update_existing,
        )
    except AdvancedStylesError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "advanced_styles_persistence_failed"},
        ) from exc
    return AdvancedStyleCsvImportResponse(imported=imported)


@router.get("/thumbnail")
def style_thumbnail(path: str) -> FileResponse:
    repository = get_repository()
    try:
        root = repository.data_dir.resolve()
        thumbnail_path = (root / path).resolve()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Style thumbnail not found") from exc
    if root not in thumbnail_path.parents or thumbnail_path.suffix.lower() != ".webp":
        raise HTTPException(status_code=404, detail="Style thumbnail not found")
    try:
        thumbnail_available = thumbnail_path.exists() and thumbnail_path.is_file()
    except OSError:
        thumbnail_available = False
    if not thumbnail_available:
        raise HTTPException(status_code=404, detail="Style thumbnail not found")
    return FileResponse(thumbnail_path)
