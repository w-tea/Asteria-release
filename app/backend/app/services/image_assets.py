from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from uuid import uuid4

from app.services.image_validation import (
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    InvalidImageContentError,
    SUPPORTED_IMAGE_FORMATS,
    validate_image_bytes,
)


class ImageAssetError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SavedImageAsset:
    path: Path
    filename: str


def safe_image_filename(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
    return name or "asset.png"


def image_asset_destination(output_dir: Path, filename: str) -> SavedImageAsset:
    safe_name = safe_image_filename(filename)
    if Path(safe_name).suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        safe_name = f"{Path(safe_name).stem or 'asset'}.png"
    final_name = f"{uuid4().hex}_{safe_name}"
    return SavedImageAsset(path=output_dir / final_name, filename=final_name)


def write_image_bytes_atomic(path: Path, data: bytes) -> None:
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
        raise ImageAssetError("save_failed") from exc


def save_validated_image_bytes(
    data: bytes,
    *,
    filename: str,
    output_dir: Path,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> SavedImageAsset:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        raise ImageAssetError("unsupported")
    if not data:
        raise ImageAssetError("empty")
    try:
        validate_image_bytes(data, suffix, max_bytes=max_bytes, max_pixels=max_pixels)
    except InvalidImageContentError as exc:
        raise ImageAssetError("invalid") from exc
    destination = image_asset_destination(output_dir, filename)
    write_image_bytes_atomic(destination.path, data)
    return destination
